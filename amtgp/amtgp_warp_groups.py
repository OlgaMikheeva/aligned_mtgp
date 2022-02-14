import tensorflow as tf
from gpflow import likelihoods
from gpflow import set_trainable


from .amtgp import AlignedMTGP


class GroupedAlignedMTGP(AlignedMTGP):
    """
    This is the fully bayesian AlignedMTGP. This version is more efficient for grouped data (warp samples
    are reused for the whole group).

    """

    def _init_warps(self, warp_groups, train_ind_loc=False, num_ind_z=10, num_ind_t=3):
        if warp_groups is None:
            warp_groups = list(range(self.num_seq))
        self.warp_groups = warp_groups
        self.warp_ids = [[] for _ in set(self.warp_groups)]
        for i in range(self.num_seq):
            self.warp_ids[self.warp_groups[i]].append(i)
        # Warps
        likelihood = likelihoods.Gaussian(variance=0.01)
        set_trainable(likelihood, False)

        list_of_kernels = isinstance(self.warp_kernel, list)

        self.G = []
        groups_ids = list(set(self.warp_groups))
        for i in range(len(groups_ids)):
            if list_of_kernels:
                kernel = self.warp_kernel[i]
            else:
                kernel = self.warp_kernel
            self.G.append(self._monotonicGP(tf.reduce_min(self.x_min), tf.reduce_max(self.x_max), kernel, likelihood,
                                            num_ind_z, num_ind_t, train_ind_loc))

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
        G_samples = [[] for _ in range(self.num_seq)]
        if num_samples is None:
            num_samples = self.path_samples
        for j in range(len(self.G)):
            g = self.G[j]
            with g.temporary_paths(num_samples=num_samples, num_bases=1024):
                for i in self.warp_ids[j]:
                    G_samples[i] = g.predict_f_samples(X_flat[rs[i]:rs[i + 1]])
        return tf.concat(G_samples, axis=1)
