from typing import Optional

import numpy as np
import tensorflow as tf
from gpflow import kernels, likelihoods
from gpflow import set_trainable
from gpflow.config import default_float
from gpflow.kernels import Kernel

from .mtgp import MTGP
from .monotonic_gp import PathwiseMonotonicSVGP
from .utils import MaybeRaggedRegressionData


class AlignedMTGP(MTGP):
    """
    This is the fully bayesian AlignedMTGP

    """

    def __init__(
        self,
        kernel_x,
        likelihood,
        inducing_variable,
        data: MaybeRaggedRegressionData,
        latent_dim: int,
        Z_data_mean: tf.Tensor,
        *,
        warp_kernel: Kernel = None,
        mean_function=None,
        q_diag: bool = False,
        q_mu=None,
        q_sqrt=None,
        whiten: bool = True,
        prior_var_z: float = 0.1,
        warp_groups=None,
        L: int = 5,
        path_samples: int = 8,
        time_in_ODE: bool = False,
    ):
        """
        - prior_kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
        - num_latent_gps is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        -L is the number of samples for the expectation of prior_kernel
        """
        self.time_in_ODE = time_in_ODE
        self.path_samples = path_samples

        super().__init__(kernel_x=kernel_x,
                         likelihood=likelihood,
                         inducing_variable=inducing_variable,
                         data=data,
                         latent_dim=latent_dim,
                         Z_data_mean=Z_data_mean,
                         mean_function=mean_function,
                         q_diag=q_diag,
                         q_mu=q_mu,
                         q_sqrt=q_sqrt,
                         whiten=whiten,
                         prior_var_z=prior_var_z,
                         L=L)

        # warps
        self.x_min = tf.reduce_min(self.X, axis=1)
        self.x_max = tf.reduce_max(self.X, axis=1)

        self.t_min = tf.reduce_min(self.X)
        self.t_max = tf.reduce_max(self.X)

        if warp_kernel is None:
            warp_kernel = kernels.Matern52(lengthscales=0.2, variance=0.01)

        self.warp_kernel = warp_kernel

        self._init_warps(warp_groups)

    def _monotonicGP(self, x_min, x_max, kern, likelihood, num_ind_z, num_ind_t, train_ind_loc):
        Z = np.linspace(x_min, x_max, num_ind_z)[:, None]
        if self.time_in_ODE:
            T = np.linspace(0., 1., num_ind_t)[:, None]
            zz, tt = np.meshgrid(Z, T, indexing='ij')
            ind_points = np.stack([np.reshape(zz, [-1]), np.reshape(tt, [-1])], axis=-1)
        else:
            ind_points = Z
        monGP = PathwiseMonotonicSVGP(kernel=kern,
                                      inducing_variable=ind_points,
                                      likelihood=likelihood,
                                      time_in_ODE=self.time_in_ODE)
        set_trainable(monGP.inducing_variable, train_ind_loc)
        return monGP

    def _init_warps(self, warp_groups, train_ind_loc=False, num_ind_z=10, num_ind_t=3):
        # Warps
        likelihood = likelihoods.Gaussian(variance=0.01)
        set_trainable(likelihood, False)

        list_of_kernels = isinstance(self.warp_kernel, list)

        if warp_groups is None:
            self.G = []
            for i in range(self.num_seq):
                if list_of_kernels:
                    kernel = self.warp_kernel[i]
                else:
                    kernel = self.warp_kernel
                self.G.append(self._monotonicGP(self.x_min[i, 0], self.x_max[i, 0], kernel, likelihood,
                                                num_ind_z, num_ind_t, train_ind_loc))

        else:
            base_G = {}
            self.G = []
            for i in range(len(warp_groups)):
                if warp_groups[i] not in base_G:
                    if list_of_kernels:
                        kernel = self.warp_kernel[warp_groups[i]]
                    else:
                        kernel = self.warp_kernel
                    base_G[warp_groups[i]] = self._monotonicGP(tf.reduce_min(self.x_min), tf.reduce_max(self.x_max),
                                                               kernel, likelihood, num_ind_z, num_ind_t, train_ind_loc)
                self.G.append(base_G[warp_groups[i]])

    def get_G_mean(self, X: tf.RaggedTensor) -> tf.Tensor:
        X_flat = X.flat_values
        rs = X.row_splits
        G_samples = []
        # TODO: do map()
        for i in range(self.num_seq):
            G_samples.append(self.G[i].predict_f(X_flat[rs[i]:rs[i + 1]]))

        G = tf.concat(G_samples, axis=0)
        G = tf.RaggedTensor.from_row_lengths(G, X.row_lengths())
        return G.flat_values

    @tf.function
    def _g_sample(self, g, x, num_samples: int = None) -> tf.Tensor:
        if num_samples is None:
            num_samples = self.path_samples
        with g.temporary_paths(num_samples=num_samples, num_bases=1024):
            return g.predict_f_samples(x)

    @tf.function
    def get_G(self, X: tf.RaggedTensor, num_samples: int = None) -> tf.Tensor:
        """
        Samples from warp posteriors
        :param num_samples: number of samples
        :param X:
        :return: tf.Tensor with dim (self.path_samples, self.num_data, 1))
        """
        X_flat = X.flat_values
        rs = X.row_splits
        G_samples = []
        # TODO: do map()
        for i in range(self.num_seq):
            G_samples.append(self._g_sample(self.G[i], X_flat[rs[i]:rs[i + 1]], num_samples))
        return tf.concat(G_samples, axis=1)

    def get_G_ragged(self, X: tf.RaggedTensor, num_samples: int = None) -> tf.Tensor:
        """
        Samples from warp posteriors
        :param num_samples: number of samples
        :param X: same input for all warp functions
        :return: tf.Tensor with dim (self.path_samples, self.num_data, 1))
        """
        G_samples = self.get_G(X, num_samples)
        G_samples = tf.transpose(G_samples, [1, 0, 2])
        return tf.RaggedTensor.from_row_lengths(G_samples, X.row_lengths())

    @tf.function
    def get_aligned_input_sample(self, X: tf.RaggedTensor, num_samples: int = None):
        """
        Prepends input data with the corresponding latent locations Z
        :return:
        """
        if num_samples is None:
            num_samples = self.path_samples
        seq_lengths = X.row_lengths(axis=1)
        Z = self.get_latent_sample(seq_lengths, num_samples)
        G = self.get_G(X, num_samples)
        return tf.concat([Z, G], axis=-1)

    def kl_g(self) -> tf.Tensor:
        return tf.reduce_sum([g.prior_kl() for g in self.G])

    def elbo(self) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, _ = self.original_data
        GZ = self.get_aligned_input_sample(X)
        _, Y = self.data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(GZ, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_mean(tf.reduce_sum(var_exp, axis=1)) * scale - kl - self.kl_z() - self.kl_g()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=default_float())])
    def _pred(self, x: tf.Tensor) -> tf.Tensor:
        samples = super().predict_f_samples(tf.transpose(x, [1, 0, 2]))
        return tf.transpose(samples, [1, 0, 2])

    def predict_f_samples(self,
                          Xnew: tf.RaggedTensor,
                          num_samples: Optional[int] = None,
                          full_cov: bool = True,
                          full_output_cov: bool = True,
                          **kwargs) -> tf.RaggedTensor:
        assert full_cov and full_output_cov, NotImplementedError

        GZ = self.get_aligned_input_sample(Xnew, num_samples)
        GZ = tf.transpose(GZ, [1, 0, 2])
        GZ_ragged = tf.RaggedTensor.from_row_lengths(GZ, Xnew.row_lengths())
        samples = tf.map_fn(self._pred, GZ_ragged)
        return samples
