from typing import Tuple, Union

import tensorflow as tf
from gpflow import kernels
from gpflow import set_trainable
from gpflow.kernels import Kernel
from gpflow.models.model import RegressionData

from amtgp.monotonic_parameter import MonotonicParameter
from amtgp.mtgp import MTGP

RaggedRegressionData = Tuple[tf.RaggedTensor, tf.RaggedTensor]
MaybeRaggedRegressionData = Union[RaggedRegressionData, RegressionData]

# types of input
SAME_X = 'same values'
SAME_LENGTH_X = 'same length'
RAGGED_X = 'ragged'


class AlignedMTGPmap(MTGP):
    """
    AlignedMTGP with MAP estimate of the warps
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
        warp_prior_kernel: Kernel = None,
        train_warps=True,
        mean_function=None,
        q_diag: bool = False,
        q_mu=None,
        q_sqrt=None,
        whiten: bool = True,
        prior_var_z: float = 0.1,
        warp_groups=None,
        L: int = 5,
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
        self.t_min = tf.reduce_min(self.X)
        self.t_max = tf.reduce_max(self.X)

        if warp_prior_kernel is None:
            warp_prior_kernel = kernels.SquaredExponential(lengthscales=0.1, variance=0.001)
            set_trainable(warp_prior_kernel.variance, False)
            set_trainable(warp_prior_kernel.lengthscales, False)

        self.warp_prior_kernel = warp_prior_kernel
        self._init_warps(warp_groups, train_warps)

    def _init_warps(self, warp_groups, train_warps, num_warp_points=100):
        # Warps
        if warp_groups is None:
            self.G = [MonotonicParameter(num=num_warp_points,
                                         prior_kernel=self.warp_prior_kernel,
                                         trainable=train_warps) for _ in range(self.num_seq)]

        else:
            base_G = {}
            self.G = []
            for i in range(len(warp_groups)):
                if warp_groups[i] not in base_G:
                    base_G[warp_groups[i]] = MonotonicParameter(num=num_warp_points,
                                                                prior_kernel=self.warp_prior_kernel,
                                                                trainable=train_warps)
                self.G.append(base_G[warp_groups[i]])

    @tf.function
    def get_G(self, X: tf.RaggedTensor) -> tf.Tensor:
        X_norm = (X - self.t_min) / (self.t_max - self.t_min)
        X_norm_flat = X_norm.flat_values
        rs = X_norm.row_splits
        Gs = [self.G[i].conditional_value(X_norm_flat[rs[i]:rs[i+1]]) for i in range(self.num_seq)]
        G = tf.concat(Gs, axis=0)
        G = tf.RaggedTensor.from_row_lengths(G, X.row_lengths())
        G = G * (self.t_max - self.t_min) + self.t_min
        return G.flat_values
