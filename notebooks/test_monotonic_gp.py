# this is example from pathwise sampling with Euler ODE solver
import time

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.config import default_float as floatx
from gpflow.utilities import print_summary, set_trainable
from gpflow.optimizers import NaturalGradient
import matplotlib.pyplot as plt
from tqdm import tqdm

from amtgp.monotonic_gp import PathwiseMonotonicSVGP


plt.rc('figure', dpi=256)
plt.rc('font', family='serif', size=12)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'''
       \usepackage{amsmath,amsfonts}
       \renewcommand{\v}[1]{\boldsymbol{#1}}''')


tf.random.set_seed(1)

t_begin = 0.
t_end = 1.
t_nsamples = 10

time_in_ODE = False

xmin = 0.1  # range over which we observe
xmax = 0.9  # the behavior of a function $f$
# X = tf.convert_to_tensor(np.linspace(xmin, xmax, 64)[:, None])

X1 = tf.convert_to_tensor(np.linspace(xmin, xmin+0.2, 50)[:, None])
X2 = tf.convert_to_tensor(np.linspace(xmax - 0.2, xmax, 50)[:, None])
X = tf.concat([X1, X2], axis=0)

Z = np.linspace(xmin, xmax, 10)[:, None]
if time_in_ODE:
    T = np.linspace(t_begin, t_end, 3)[:, None]

    zz, tt = np.meshgrid(Z, T, indexing='ij')
    ind_points = np.stack([np.reshape(zz, [-1]), np.reshape(tt, [-1])], axis=-1)
else:
    ind_points = Z

kernel = gpflow.kernels.Matern52(lengthscales=0.2, variance=0.01)
likelihood = gpflow.likelihoods.Gaussian(variance=0.01)

model = PathwiseMonotonicSVGP(kernel=kernel,
                              likelihood=likelihood,
                              inducing_variable=ind_points,
                              time_in_ODE=time_in_ODE)

# Generate toy data
noise = tf.constant(0.0001, dtype=floatx())

f = 0.1 * tf.sin(X*10) + X
y = f + tf.random.normal(stddev=tf.sqrt(noise), shape=tf.shape(f), dtype=floatx())

# Keep things sane since we don't have a lot of data,
gpflow.utilities.set_trainable(model.inducing_variable, False)
# gpflow.utilities.set_trainable(model.likelihood.variance, False)

print_summary(model)

# Sample-based training
num_steps = 50
step_sizes = [1e-1, 1e-2, 1e-3]
boundaries = [k * num_steps // len(step_sizes) for k in range(1, len(step_sizes))]
schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, step_sizes)
optimizer = tf.keras.optimizers.Adam(schedule)

training_loss = model.training_loss_closure(data=(X, y), compile=True)
trainables = model.trainable_variables
step_iterator = tqdm(range(num_steps))

train_with_natgrad = False
variational_params = [(model.q_mu, model.q_sqrt)]
natgrad_opt = NaturalGradient(gamma=.1)

if train_with_natgrad:
    set_trainable(model.q_mu, False)
    set_trainable(model.q_sqrt, False)

times = []
for step in step_iterator:
    if train_with_natgrad:
        natgrad_opt.minimize(training_loss, var_list=variational_params)
    start_time = time.time()
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(trainables)
        loss = training_loss()

    grads = tape.gradient(loss, trainables)
    grads_and_vars = tuple(zip(grads, trainables))
    optimizer.apply_gradients(grads_and_vars)
    end_time = time.time()
    loss_ema = loss if (step == 0) else loss_ema + 0.25 * (loss - loss_ema)
    step_iterator.set_postfix_str(f'EMA(loss): {loss_ema:.3e}')

print_summary(model)

# Sample posterior
Xnew = tf.linspace(tf.cast(0.0, floatx()), tf.cast(1.0, floatx()), 100)
mu, sigma2 = map(tf.squeeze, model.predict_f(Xnew[:, None]))

with model.temporary_paths(num_samples=1000, num_bases=1024) as temp_paths:
    fnew = tf.squeeze(model.predict_f_samples(Xnew[:, None]))
pnew = fnew

fig, ax = plt.subplots(1, 1, figsize=(7, 6))
lower = tf.cast(0.025, floatx())
upper = tf.cast(0.975, floatx())

# Visualize the training data
ax.scatter(X, y,
              zorder=9999,  # place these on top
              s=16,
              color='tab:blue',
              linewidth=0.75,
              edgecolor='k'
              )

# Show empirical quantiles
ax.fill_between(Xnew,
                *np.quantile(pnew, q=(lower, upper), axis=0),
                color='tab:blue',
                alpha=0.15)

ax.plot(Xnew, mu, '--', color='r')

# Visualize some sample paths
for probs in pnew[:16]:
    ax.plot(Xnew, probs, alpha=0.5, linewidth=0.5, color='tab:blue')

# Format axes
_ = ax.set_xlabel(r'$\v{x} \in \mathbb{R}$')
_ = ax.set_ylabel(r'$g(\v{x})$')

plt.show()