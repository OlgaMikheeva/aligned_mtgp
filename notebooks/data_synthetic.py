import numpy as np
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import gpflow
from gpflow.config import default_float, default_jitter


def _generate_warps(n, x, seed):
    G = []
    tf.random.set_seed(seed)
    warp_kernel = gpflow.kernels.SquaredExponential(lengthscales=1., variance=.1)
    for i in range(n):
        K = warp_kernel.K(x)
        K += default_jitter() * tf.eye(x.shape[0], dtype=default_float())
        dist = tfd.MultivariateNormalTriL(loc=x[:, 0], scale_tril=tf.linalg.cholesky(K))
        s = dist.sample(1)
        G.append(s.numpy().T)
    tf.random.set_seed(None)
    return G


def generate_synthetic_data():
    x = np.expand_dims(np.linspace(0, 10, num=100), axis=-1)
    S = 5
    noise_std = 0.05
    np.random.seed(123)
    G = _generate_warps(n=S * 2, x=x, seed=3)
    X = [x] * len(G)

    def f1(y): return np.sin(y)

    def f2(y): return (np.sin((y - 2) / 2) + np.sin((y * 2))) / 2

    F = list(map(f1, G[:S])) + list(map(f2, G[S:]))
    Y = [f + noise_std * np.random.normal(size=f.shape) for f in F]
    np.random.seed(None)
    return X, Y, F, G
