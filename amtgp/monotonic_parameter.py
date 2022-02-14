from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability.python.bijectors as tfb  # workaround for PyCharm bug
import tensorflow_probability.python.distributions as tfd
from gpflow.kernels import Kernel
from gpflow import kernels
from gpflow.base import Parameter
from gpflow.conditionals import conditional
from gpflow.utilities import to_default_float
from gpflow.config import default_float, default_jitter
from gpflow.base import PriorOn


class MonotonicParameter(Parameter):
    def __init__(
        self,
        num: int,
        prior_kernel: Kernel,
        trainable: bool = True
    ):
        # TODO: update the description
        """

        """
        u = np.linspace(0., 1., num=num, endpoint=True)[1:]
        u[0] += 1e-6
        value = tf.convert_to_tensor(u, dtype=default_float())

        shift = Parameter(0., dtype=default_float(), transform=tfb.SoftClip(low=to_default_float(-1.),
                                                                            high=to_default_float(0.01)),
                          name='G_shift', trainable=trainable)  # [-1., 0.]
        scale = Parameter(1., dtype=default_float(), transform=tfb.SoftClip(low=to_default_float(.99),
                                                                            high=to_default_float(2.)),
                          name='G_scale', trainable=trainable)  # [1., 2.]

        monotonic_transform = tfb.Chain([
            tfb.Shift(shift=shift),
            tfb.Scale(scale=scale),
            tfb.Cumsum(),
            tfb.SoftmaxCentered()])

        self.K = kernels.SquaredExponential(lengthscales=.1, variance=0.1)
        # GP prior with Identity mean
        self.x = np.linspace(0., 1., num=num, endpoint=True)
        self.K_prior = prior_kernel
        cov = self.K_prior.K(tf.expand_dims(self.x, axis=-1))
        cov += default_jitter() * tf.eye(num, dtype=default_float())
        prior = tfd.MultivariateNormalTriL(loc=self.x, scale_tril=tf.linalg.cholesky(cov))

        super().__init__(value=value, transform=monotonic_transform, prior=prior, prior_on=PriorOn.CONSTRAINED,
                         trainable=trainable, dtype=default_float(), name='U')

        self.u_prior = tfd.Normal(loc=to_default_float(0), scale=to_default_float(1.))

    def value(self):
        return self.transform(self.unconstrained_variable)

    def log_prior_density(self) -> tf.Tensor:
        """ Log of the prior probability density of the constrained variable. """

        if self.prior is None:
            return tf.convert_to_tensor(0.0, dtype=self.dtype)

        # GP prior
        y = self.conditional_value()
        gp_log_prob = tf.reduce_sum(self.prior.log_prob(y))

        # TODO: add unconditional prior for values to be close to 0 (so that it's smoother
        #  and less likely to break monotonicity)
        # unconstrained prior
        x = self.unconstrained_variable
        u_log_prob = tf.reduce_sum(self.u_prior.log_prob(x))
        return gp_log_prob + u_log_prob

    def conditional_value(self, x_new: tf.Tensor = None) -> tf.Tensor:
        if x_new is None:
            x_new = to_default_float(self.x)
            x_new = tf.expand_dims(x_new, axis=-1)
        x = tf.expand_dims(to_default_float(self.x[1:]), axis=-1)
        f = tf.expand_dims(self.value(), axis=-1) - x
        mu, _ = conditional(
            x_new,
            x,
            self.K,
            f)
        return tf.squeeze(mu + x_new, axis=-1)

    def posterior(self, x_train: tf.Tensor,  x_new: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        mu_train = self.conditional_value(x_train)
        mu_train = tf.expand_dims(mu_train, axis=-1) - x_train  # remove the identity mean
        mu, cov = conditional(
            x_new,
            x_train,
            self.K,
            mu_train,
            full_cov=True)
        return tf.squeeze(mu + x_new, axis=-1), cov
