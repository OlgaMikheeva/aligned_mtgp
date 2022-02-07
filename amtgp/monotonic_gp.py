from typing import Optional

import numpy as np
import tensorflow as tf
from gpflow_sampling.models import PathwiseSVGP
from gpflow.base import TensorLike
from gpflow import posteriors
from gpflow.models.model import InputData, MeanAndVariance

from .euler import EulerODE


class PathwiseMonotonicSVGP(PathwiseSVGP):
    """
    ODE-based monotonic GP with pathwise sampling
    """
    def __init__(self, *args,
                 time_in_ODE: bool = False,
                 t_begin: float = 0.,
                 t_end: float = 1.,
                 t_nsamples: int = 10,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.time_in_ODE = time_in_ODE
        self.t_begin = t_begin
        self.t_end = t_end
        self.t_nsamples = t_nsamples

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        if self.time_in_ODE:
            T = np.linspace(self.t_begin, self.t_end, 10)[:, None]
            xx, tt = np.meshgrid(Xnew, T, indexing='ij')
            XTnew = tf.stack([tf.reshape(xx, [-1]), tf.reshape(tt, [-1])], axis=-1)
            _, var = super().predict_f(XTnew, full_cov=full_cov, full_output_cov=full_output_cov)
        else:
            _, var = super().predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)

        def ode_fn(t, x):
            if self.time_in_ODE:
                t_ = tf.tile(tf.reshape(t, [1, 1]), [tf.shape(x)[0], 1])
                x_ = tf.concat([x, t_], axis=-1)
            else:
                x_ = x
            m, _ = self.posterior(posteriors.PrecomputeCacheType.NOCACHE).fused_predict_f(
                x_, full_cov=False, full_output_cov=False)
            return m

        ode_solver = EulerODE(ode_fn, self.t_end, self.t_nsamples)
        f, _ = ode_solver.forward(Xnew, save_intermediate=False)
        return f, var

    def predict_f_samples(self,
                          Xnew: TensorLike,
                          num_samples: Optional[int] = None,
                          full_cov: bool = True,
                          full_output_cov: bool = True,
                          **kwargs) -> tf.Tensor:
        assert full_cov and full_output_cov, NotImplementedError
        if self.paths is None:
            raise RuntimeError("Paths were not initialized.")
        if num_samples is not None:
            assert num_samples == self.paths.sample_shape, \
                ValueError("Requested number of samples does not match path count.")

        X_tiled = tf.tile(tf.expand_dims(Xnew, axis=0), [self.paths.sample_shape[0], 1, 1])

        def ode_fn(t, x):
            if self.time_in_ODE:
                t_ = tf.tile(tf.reshape(t, [1, 1, 1]), [self.paths.sample_shape[0], tf.shape(x)[1], 1])
                x_ = tf.concat([x, t_], axis=-1)
            else:
                x_ = x
            return self.paths(x_, sample_axis=0, **kwargs)

        ode_solver = EulerODE(ode_fn, self.t_end, self.t_nsamples)
        f, _ = ode_solver.forward(X_tiled, save_intermediate=False)
        return f

    def elbo(self, data: tuple, num_samples: int = 32, num_bases: int = 1024) -> tf.Tensor:
        """
        Estimate the evidence lower bound on the log marginal likelihood of the model
        by using decoupled sampling to construct a Monte Carlo integral.
        """
        X, y = data
        with self.temporary_paths(num_samples=num_samples, num_bases=num_bases):
            f = self.predict_f_samples(X)
        kl = self.prior_kl()
        monte_carlo = tf.reduce_mean(self.likelihood.log_prob(f, y), axis=0)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)

        return tf.reduce_sum(monte_carlo) * scale - kl