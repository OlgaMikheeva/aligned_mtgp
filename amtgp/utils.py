import tensorflow as tf
from gpflow.utilities.ops import pca_reduce


def maybe_ragged_pca(Y, latent_dim):
    # PCA should be done on truncated sequences, because they can be different size
    Y_for_pca = Y
    num_seq = Y.shape[0]
    if isinstance(Y, tf.RaggedTensor):
        seq_lengths = Y.row_lengths(axis=1)
        min_length = tf.reduce_min(seq_lengths)
        output_dim = Y.shape[2]
        Y_for_pca = Y.to_tensor(shape=(num_seq, min_length, output_dim))
    Y_for_pca = tf.reshape(Y_for_pca, (num_seq, -1))
    return pca_reduce(Y_for_pca, latent_dim)
