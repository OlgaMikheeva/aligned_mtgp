from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
from gpflow import kernels, likelihoods
from gpflow import set_trainable
from gpflow.base import Parameter
from gpflow.conditionals import conditional, uncertain_conditional
from gpflow.config import default_float
from gpflow.kernels import Kernel
from gpflow.mean_functions import Zero
from gpflow.models.model import GPModel, MeanAndVariance, RegressionData
from gpflow.models.training_mixins import InputData, InternalDataTrainingLossMixin
from gpflow.utilities import positive, triangular
from gpflow import kullback_leiblers
from gpflow.models.util import inducingpoint_wrapper

from .monotonic_gp import PathwiseMonotonicSVGP

from gpflow.models.gplvm import BayesianGPLVM

RaggedRegressionData = Tuple[tf.RaggedTensor, tf.RaggedTensor]
MaybeRaggedRegressionData = Union[RaggedRegressionData, RegressionData]

# types of input
SAME_X = 'same values'
SAME_LENGTH_X = 'same length'
RAGGED_X = 'ragged'


class AlignedMTGP(GPModel, InternalDataTrainingLossMixin):
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
        X, Y = data
        self.original_data = data
        self.num_seq = Y.shape[0]
        output_dim = Y.shape[2]
        self.input_dim = X.shape[2]
        self.latent_dim = latent_dim
        self.L = L
        self.time_in_ODE = time_in_ODE
        self.path_samples = path_samples

        # check that input and output shapes match
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
        self.x_min = tf.reduce_min(X, axis=1)
        self.x_max = tf.reduce_max(X, axis=1)

        self.t_min = tf.reduce_min(X)
        self.t_max = tf.reduce_max(X)

        if warp_kernel is None:
            warp_kernel = kernels.Matern52(lengthscales=0.2, variance=0.01)

        self.warp_kernel = warp_kernel

        self._init_warps(warp_groups)

    def _init_warps(self, warp_groups, train_ind_loc=False, num_ind_z=10, num_ind_t=3):
        # Warps
        likelihood = likelihoods.Gaussian(variance=0.01)
        set_trainable(likelihood, False)

        list_of_kernels = isinstance(self.warp_kernel, list)

        def monotonicGP(x_min, x_max, kern):
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
        if warp_groups is None:
            self.G = []
            for i in range(self.num_seq):
                if list_of_kernels:
                    kernel = self.warp_kernel[i]
                else:
                    kernel = self.warp_kernel
                self.G.append(monotonicGP(self.x_min[i, 0], self.x_max[i, 0], kernel))

        else:
            base_G = {}
            self.G = []
            for i in range(len(warp_groups)):
                if warp_groups[i] not in base_G:
                    if list_of_kernels:
                        kernel = self.warp_kernel[warp_groups[i]]
                    else:
                        kernel = self.warp_kernel
                    base_G[warp_groups[i]] = monotonicGP(tf.reduce_min(self.x_min), tf.reduce_max(self.x_max), kernel)
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

    def prior_kl(self) -> tf.Tensor:
        return kullback_leiblers.prior_kl(
            self.inducing_variable, self.kernel, self.q_mu, self.q_sqrt, whiten=self.whiten
        )

    def kl_z(self) -> tf.Tensor:
        q_z = tfd.Normal(loc=self.Z_mean, scale=tf.sqrt(self.Z_var))
        p_z = tfd.Normal(loc=self.Z_prior_mean, scale=tf.sqrt(self.Z_prior_var))
        return tf.reduce_sum(q_z.kl_divergence(p_z))

    def kl_g(self) -> tf.Tensor:
        return tf.reduce_sum([g.prior_kl() for g in self.G])

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()

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
