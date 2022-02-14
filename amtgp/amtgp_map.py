from typing import Tuple, Union

import numpy as np
import tensorflow as tf
from gpflow import kernels
from gpflow import set_trainable
from gpflow.base import Parameter
from gpflow.conditionals import conditional
from gpflow.config import default_float
from gpflow.kernels import Kernel
from gpflow.mean_functions import Zero
from gpflow.models.model import GPModel, MeanAndVariance, RegressionData
from gpflow.models.training_mixins import InputData, InternalDataTrainingLossMixin
from gpflow.utilities import positive, triangular
from gpflow import kullback_leiblers
from gpflow.models.util import inducingpoint_wrapper
import tensorflow_probability.python.distributions as tfd  # workaround for PyCharm bug

from amtgp.monotonic_parameter import MonotonicParameter


RaggedRegressionData = Tuple[tf.RaggedTensor, tf.RaggedTensor]
MaybeRaggedRegressionData = Union[RaggedRegressionData, RegressionData]

# types of input
SAME_X = 'same values'
SAME_LENGTH_X = 'same length'
RAGGED_X = 'ragged'


class AlignedMTGPmap(GPModel, InternalDataTrainingLossMixin):
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
        X, Y = data
        self.original_data = data
        self.num_seq = Y.shape[0]
        output_dim = Y.shape[2]
        self.input_dim = X.shape[2]
        self.latent_dim = latent_dim
        self.L = L

        assert type(X) == type(Y), 'input output type mismatch'

        if isinstance(X, tf.RaggedTensor):
            self.type_of_X = RAGGED_X
            self.seq_lengths = X.row_lengths(axis=1)
            self.num_data = X.flat_values.shape[0]
            X_flat = X.flat_values
            Y_flat = Y.flat_values
        else:
            self.seq_lengths = X.shape[1]
            self.num_data = self.seq_lengths * self.num_seq
            if X.shape[0] == 1:
                self.type_of_X = SAME_X
                X_flat = tf.reshape(tf.repeat(X, repeats=[self.num_seq], axis=0), (-1, self.input_dim))
            else:
                self.type_of_X = SAME_LENGTH_X
                X_flat = tf.reshape(X, (-1, self.input_dim))
            Y_flat = tf.reshape(X, (-1, output_dim))
        self.data = X_flat, Y_flat

        # latent variable
        self.Z_mean = Parameter(Z_data_mean)
        self.Z_var = Parameter(0.1 * tf.ones((self.num_seq, self.latent_dim),
                                             dtype=default_float()),
                               transform=positive())

        # parameters for the prior mean variance of Z
        Z_prior_mean = tf.zeros((self.num_seq, self.latent_dim), dtype=default_float())
        Z_prior_var = prior_var_z * tf.ones((self.num_seq, self.latent_dim), dtype=default_float())
        self.Z_prior_mean = tf.convert_to_tensor(np.atleast_1d(Z_prior_mean), dtype=default_float())
        self.Z_prior_var = tf.convert_to_tensor(np.atleast_1d(Z_prior_var), dtype=default_float())

        if Z_data_mean.shape[1] != latent_dim:
            msg = "Passed in number of latent {0} does not match initial X {1}."
            raise ValueError(msg.format(latent_dim, Z_data_mean.shape[1]))

        if mean_function is None:
            mean_function = Zero()

        # check prior_kernel and set default prior_kernel
        kernel_z = kernels.SquaredExponential(lengthscales=1.,
                                              active_dims=list(range(0, latent_dim)), name='kernel_Z')
        if kernel_x is None:
            kernel_x = kernels.SquaredExponential(lengthscales=1.,
                                                  active_dims=list(range(latent_dim, latent_dim + self.input_dim)),
                                                  name='kernel_X')
        # kernel_x.variance.prior = tfd.LogNormal(loc=tf.constant(0., dtype=default_float()),
        #                                         scale=tf.constant(1., dtype=default_float()))

        kernel = kernels.Product([kernel_z, kernel_x])
        set_trainable(kernel_z.variance, False)
        set_trainable(kernel_z.lengthscales, False)

        # init the super class, accept args
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=output_dim)
        self.q_diag = q_diag
        self.whiten = whiten

        # initialize inducing variables
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        if not self.inducing_variable.Z.shape[1] == latent_dim + self.input_dim:
            raise ValueError("Second dim of the inducing points should have size = latent_dim + input_dim")

        # init variational parameters
        num_inducing = len(self.inducing_variable)
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)

        # warps
        self.t_min = tf.reduce_min(X)
        self.t_max = tf.reduce_max(X)

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

    def print_data_info(self):
        print('\n########## DATA INFO ##########')
        print('Type of input:       ', self.type_of_X)
        print('Number of sequences: ', self.num_seq)
        seq_lengths = self.seq_lengths.numpy() if isinstance(self.seq_lengths, tf.Tensor) else self.seq_lengths
        print('Lengths of sequences:', seq_lengths)
        print('Total number of points:', self.num_data)
        print('Input dims:', self.input_dim)
        print('Output dims:', self.num_latent_gps)
        print('Latent dims:', self.latent_dim)
        print('\n')

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.

        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.

        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros((num_inducing, self.num_latent_gps)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]

        if q_sqrt is None:
            if self.q_diag:
                ones = np.ones((num_inducing, self.num_latent_gps), dtype=default_float())
                self.q_sqrt = Parameter(ones, transform=positive())  # [M, P]
            else:
                q_sqrt = [
                    np.eye(num_inducing, dtype=default_float()) for _ in range(self.num_latent_gps)
                ]
                q_sqrt = np.array(q_sqrt)
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [P, M, M]
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent_gps = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=positive())  # [M, L|P]
            else:
                assert q_sqrt.ndim == 3
                self.num_latent_gps = q_sqrt.shape[0]
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [L|P, M, M]

    def get_latent_sample(self, row_lengths, L):
        """
        Constructs latent coordinate for each data point by repeating the latent coordinate for the
        corresponding sequence
        :return: tf.Tensor
        """
        q_z = tfd.Normal(loc=self.Z_mean, scale=tf.sqrt(self.Z_var))
        Z = q_z.sample(L)
        return tf.repeat(Z, repeats=row_lengths, axis=1)

    def get_latent_means(self, row_lengths):
        """
        Constructs latent mean coordinate for each data point by repeating the latent coordinate for the
        corresponding sequence
        :return: tf.Tensor
        """
        return tf.repeat(self.Z_mean, repeats=row_lengths, axis=0)

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

    @tf.function
    def get_aligned_data_sample(self):
        """
        Prepends input data with the corresponding latent locations Z
        :return:
        """
        Z = self.get_latent_sample(self.seq_lengths, self.L)
        _, Y = self.data
        X, _ = self.original_data
        G = tf.expand_dims(self.get_G(X), axis=-1)
        G = tf.tile(tf.expand_dims(G, axis=0), multiples=[self.L, 1, 1])
        return tf.concat([Z, G], axis=-1), Y

    def get_aligned_input(self, X):
        """
        Prepends input data with the corresponding latent locations Z
        :return:
        """
        Z = self.get_latent_means(X.row_lengths())
        G = tf.expand_dims(self.get_G(X), axis=-1)
        return tf.concat([Z, G], axis=1)

    def get_aligned_input_sample(self, X, L: int = None):
        """
        Prepends input data with the corresponding latent locations Z
        :return:
        """
        if L is None:
            L = self.L
        Z = self.get_latent_sample(X.row_lengths(), L)
        G = tf.expand_dims(self.get_G(X), axis=-1)
        G = tf.tile(tf.expand_dims(G, axis=0), multiples=[L, 1, 1])
        return tf.concat([Z, G], axis=-1)

    def prior_kl(self) -> tf.Tensor:
        return kullback_leiblers.prior_kl(
            self.inducing_variable, self.kernel, self.q_mu, self.q_sqrt, whiten=self.whiten
        )

    def kl_z(self) -> tf.Tensor:
        q_z = tfd.Normal(loc=self.Z_mean, scale=tf.sqrt(self.Z_var))
        p_z = tfd.Normal(loc=self.Z_prior_mean, scale=tf.sqrt(self.Z_prior_var))
        return tf.reduce_sum(q_z.kl_divergence(p_z))

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()

    def elbo(self) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = self.get_aligned_data_sample()
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_mean(tf.reduce_sum(var_exp, axis=1)) * scale - kl - self.kl_z()

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        mu, var = conditional(
            Xnew,
            self.inducing_variable,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        return mu, var
