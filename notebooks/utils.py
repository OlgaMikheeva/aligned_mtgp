from typing import List
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
from gpflow.utilities import to_default_float
from gpflow.ci_utils import ci_niter
from gpflow import set_trainable
from gpflow.optimizers import NaturalGradient
from gpflow.config import default_float

from amtgp.utils import maybe_ragged_pca


def initialize_z(data, *, PCA_init=True, latent_dim=2):
    # PCA to initialize latent Z
    if PCA_init:
        Z_pca = maybe_ragged_pca(data[1], latent_dim)
        Z_init = (Z_pca - tf.reduce_mean(Z_pca)) / tf.math.reduce_std(Z_pca)
    else:
        Z_init = to_default_float(np.random.normal(size=(data[1].shape[0], latent_dim)))
    return Z_init


def initialize_inducing_points(data, *, num_ind, Z_init):
    # Initialize inducing locations to M random inputs in the dataset (with added Z_pca component)
    row_lengths = data[0].row_lengths().numpy()
    total_size = np.sum(row_lengths)
    ind_idx = np.random.choice(total_size, size=num_ind, replace=False)
    Z_extended = np.repeat(Z_init, repeats=row_lengths, axis=0)
    input_extended = np.concatenate([Z_extended, data[0].flat_values], axis=1)
    return input_extended[ind_idx, :].copy()


def run_adam(model,
             iterations,
             train_with_natgrad,
             step_var: tf.Variable,
             manager: tf.train.CheckpointManager,
             is_warp_nat_grad=False,
             save_step=100,
             print_step: int = 100,
             nat_grad_gamma=0.5,
             warp_nat_grad_gamma=0.05):
    """
    Utility function running the Adam optimizer

    :param warp_nat_grad_gamma:
    :param nat_grad_gamma:
    :param print_step:
    :param save_step:
    :param is_warp_nat_grad:
    :param manager:
    :param step_var:
    :param iterations:
    :param train_with_natgrad:
    :param model: GPflow model
    """
    # Create an Adam Optimizer action
    training_loss = model.training_loss_closure(compile=True)
    optimizer = tf.optimizers.Adam(learning_rate=0.005)
    variational_params = [(model.q_mu, model.q_sqrt)]
    natgrad_opt = NaturalGradient(gamma=nat_grad_gamma)
    if is_warp_nat_grad:
        natgrad_opt2 = NaturalGradient(gamma=warp_nat_grad_gamma)
        variational_params2 = []
    if train_with_natgrad:
        set_trainable(model.q_mu, False)
        set_trainable(model.q_sqrt, False)
        if is_warp_nat_grad:
            for G in model.G:
                variational_params2.append((G.q_mu, G.q_sqrt))
                set_trainable(G.q_mu, False)
                set_trainable(G.q_sqrt, False)

    @tf.function
    def optimization_step():
        if train_with_natgrad:
            natgrad_opt.minimize(training_loss, var_list=variational_params)
            if is_warp_nat_grad:
                natgrad_opt2.minimize(training_loss, var_list=variational_params2)
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        step_var.assign_add(1)
        if step % print_step == 0:
            loss = training_loss().numpy()
            elbo = model.elbo().numpy()
            print(step_var.numpy(), 'loss:', loss, 'ELBO:', elbo)
        if manager is not None and step % save_step == 0:
            manager.save()


def train(model, it, train_with_natgrad,
          step_var=None,
          manager=None,
          is_warp_nat_grad=False,
          save_step=200,
          print_step=100,
          nat_grad_gamma=0.5,
          warp_nat_grad_gamma=0.05):
    maxiter = ci_niter(it)
    s = time.time()
    if step_var is None:
        step_var = tf.Variable(1, dtype=tf.int32, trainable=False)
    run_adam(model, maxiter, train_with_natgrad, step_var, manager,
             is_warp_nat_grad=is_warp_nat_grad,
             save_step=save_step,
             print_step=print_step,
             nat_grad_gamma=nat_grad_gamma,
             warp_nat_grad_gamma=warp_nat_grad_gamma
             )
    print('time to train:', time.time() - s)
    print('elbo:', model.elbo().numpy())


def make_ragged_tensor(x: List[np.ndarray]):
    x_flat = np.concatenate(x, axis=0)
    return tf.RaggedTensor.from_row_lengths(to_default_float(x_flat), [xx.shape[0] for xx in x])


def remove_data_random(x, y, missing_ratio, cont_segment=False, same_loc=False, remove_seq=None, return_loc=False):
    X_obs = []
    Y_obs = []
    X_missing = []
    Y_missing = []
    same_start = np.random.choice(range(0, x[0].shape[0] - int(x[0].shape[0] * missing_ratio)), size=1)[0]

    loc = []
    if remove_seq is None:
        remove_seq = list(range(len(x)))
    for i in range(len(x)):
        if i in remove_seq:
            num_missing = int(x[i].shape[0] * missing_ratio)
        else:
            num_missing = 0
        num_keep = x[i].shape[0] - num_missing
        if cont_segment:
            if same_loc:
                start = same_start
            else:
                start = np.random.choice(range(0, x[i].shape[0] - num_missing), size=1)[0]
            end = start + num_missing
            if return_loc:
                loc.append([x[i][start, :], x[i][end, :]])
            x_miss = x[i][start:end, :]
            y_miss = y[i][start:end, :]
            if start == 0:
                x_obs = x[i][end:, :]
                y_obs = y[i][end:, :]
            else:
                x_obs = np.concatenate([x[i][:start, :], x[i][end:, :]], axis=0)
                y_obs = np.concatenate([y[i][:start, :], y[i][end:, :]], axis=0)
        else:
            ind_to_keep = np.random.choice(x[i].shape[0], size=num_keep, replace=False)
            ind_to_keep.sort()
            ind_to_remove = np.array([j for j in range(x[i].shape[0]) if j not in ind_to_keep])
            x_obs = x[i][ind_to_keep, :]
            y_obs = y[i][ind_to_keep, :]
            x_miss = x[i][ind_to_remove, :]
            y_miss = y[i][ind_to_remove, :]
        X_obs.append(x_obs)
        Y_obs.append(y_obs)
        X_missing.append(x_miss)
        Y_missing.append(y_miss)
    X_missing = make_ragged_tensor(X_missing)
    Y_missing = make_ragged_tensor(Y_missing)
    data_miss = X_missing, Y_missing
    X_obs = make_ragged_tensor(X_obs)
    Y_obs = make_ragged_tensor(Y_obs)
    data = X_obs, Y_obs
    if return_loc:
        return data, data_miss, loc
    return data, data_miss


def smse(Y_miss, Y_pred_mean):
    var = tf.math.reduce_variance(Y_miss.flat_values)
    mse = tf.reduce_mean(tf.square(Y_miss.flat_values - Y_pred_mean))
    return mse / var


def mean_pred(model, X_miss):
    GZ = model.get_aligned_input(X_miss)
    GZ = tf.expand_dims(GZ, axis=0)
    mean, _ = model.predict_f(GZ, full_cov=False)
    return mean


def mean_pred2(model, X_miss, num_samples=100):
    GZ = model.get_aligned_input_sample(X_miss, num_samples=num_samples)
    means, _ = model.predict_f(GZ, full_cov=False, full_output_cov=False)
    mean = tf.reduce_mean(means, axis=0)
    return mean


def _baseline_log_prob(Ytrain, Ypred):
    # baseline
    mean = np.mean(Ytrain, axis=0)
    std = np.std(Ytrain, axis=0)
    dist = tfd.Normal(loc=mean, scale=std)
    return dist.log_prob(Ypred.flat_values)[:, 0]


def snlp(model, Xpred, Ypred, num_samples=100):
    GZ = model.get_aligned_input_sample(Xpred, num_samples)
    log_density = model.predict_log_density((GZ, Ypred.flat_values), full_cov=False)
    log_density = tf.reduce_mean(log_density, axis=0)
    # baseline
    Ytrain = model.data[1].numpy()
    base_log_prob = _baseline_log_prob(Ytrain, Ypred)
    return - np.sum(log_density - base_log_prob)


def snlp2(model, Xpred, Ypred, num_samples=100):
    log_density = model.predict_log_exp_density((Xpred, Ypred), num_samples=num_samples, full_cov=False)
    # baseline
    Ytrain = model.data[1].numpy()
    base_log_prob = _baseline_log_prob(Ytrain, Ypred)
    return - np.sum(log_density - base_log_prob)


def get_latent_space(mean, std, z_min=-1, z_max=1, n=100):
    J = mean.shape[0]
    dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)
    mix_dist = tfd.MixtureSameFamily(tfd.Categorical(probs=tf.ones(J, dtype=default_float())/J),
                                                   dist)

    z1 = np.linspace(z_min, z_max, n)
    z2 = np.linspace(z_min, z_max, n)
    z1m, z2m = np.meshgrid(z1, z2)
    z = np.concatenate([np.reshape(z1m, (-1, 1)), np.reshape(z2m, (-1, 1))], axis=-1)
    z = np.expand_dims(z, axis=0)
    z = np.transpose(z, axes=[1, 0, 2])
    pz = mix_dist.log_prob(z)
    pz = np.reshape(pz, (n, n))
    return z1m, z2m, pz