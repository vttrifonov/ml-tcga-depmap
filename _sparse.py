import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np

class Sparse(tfk.layers.Layer):
    def __init__(self, ij):
        super().__init__()
        u = np.unique(ij[:,0], return_index=True)
        u = u[1][1:]
        self.kj = (
            tf.ragged.constant(np.split(np.arange(ij.shape[0]), u)),
            tf.ragged.constant(np.split(ij[:,1], u))
         )
        self.sparse_kernel = self.add_weight(
            name='sparse_kernel',
            shape=(ij.shape[0],1),
            trainable=True
        )

    def call(self, inputs):
        def _mult(k, j):
            kernel = tf.gather(self.sparse_kernel, k, axis=0)
            input = tf.gather(inputs, j, axis=1)
            result = tf.tensordot(input, kernel, 1)
            return result

        outputs = tf.map_fn(lambda x: _mult(*x), self.kj, fn_output_signature=tf.float32)
        outputs = tf.reshape(outputs, (self.kj[0].shape[0], tf.shape(inputs)[0]))
        outputs = tf.transpose(outputs)
        return outputs

class Sparse1(tfk.layers.Layer):
    def __init__(self, ij, dense_shape):
        super().__init__()
        self.ij = ij
        self.dense_shape= dense_shape
        self.kernel = self.add_weight(
            name='sparse_kernel',
            shape=(ij.shape[0],),
            trainable=True
        )

    def call(self, inputs):
        sparse = tf.SparseTensor(
            indices = self.ij,
            values = self.kernel,
            dense_shape = self.dense_shape
        )
        outputs = tf.sparse.sparse_dense_matmul(sparse, inputs, adjoint_b=True)
        outputs = tf.transpose(outputs)
        return outputs

class Sparse2(tfk.layers.Layer):
    def __init__(self, ij, dense_shape):
        super().__init__()
        self.ij = ij
        self.dense_shape= dense_shape
        self.kernel = self.add_weight(
            name='kernel',
            shape=(ij.shape[0],),
            trainable=True
        )
        self.ids = tf.SparseTensor(
            indices = ij,
            values = ij[:,1],
            dense_shape = dense_shape
        )

    def call(self, inputs):
        weights = tf.SparseTensor(
            indices = self.ij,
            values = self.kernel,
            dense_shape = self.dense_shape
        )
        outputs = tf.transpose(inputs)
        outputs = tf.nn.embedding_lookup_sparse(outputs, self.ids, weights, combiner='sum')
        outputs = tf.transpose(outputs)
        outputs = tf.reshape(outputs, (tf.shape(inputs)[0], self.dense_shape[0]))
        return outputs
