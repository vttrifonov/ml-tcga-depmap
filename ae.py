import sklearn.decomposition as skld
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow as tf

class PCA:
    def __init__(
        self,
        encoding_dim
    ):
        self.encoding_dim = encoding_dim

        self.model = skld.PCA(n_components=encoding_dim)

        self.encode = self.model.transform
        self.decode = self.model.inverse_transform

class AE:
    def __init__(
        self,
        input_dim, encoding_dim,
        encoder_activation,
        decoder_activation,
        optim, loss
    ):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        encoder = keras.Sequential([
            layers.Input((self.input_dim,)),
            layers.Dense(self.encoding_dim, activation=encoder_activation)
        ])

        decoder = keras.Sequential([
            layers.Input((self.encoding_dim,)),
            layers.Dense(self.input_dim, activation=decoder_activation)
        ])

        autoencoder = keras.Model(encoder.input, decoder(encoder.output))
        autoencoder.compile(optimizer=optim, loss=loss)

        self.model = autoencoder
        self.encoder = encoder
        self.decoder = decoder

        self.encode = encoder.predict
        self.decode = decoder.predict


class AE1:
    def __init__(
        self,
        input_dim,
        encoder_dims,
        decoder_dims
    ):
        self.input_dim = input_dim
        self.latent_dim = encoder_dims[-1]

        encoder = keras.Sequential()
        encoder.add(layers.Input(shape=(self.input_dim,)))
        for dim in encoder_dims:
            encoder.add(layers.Dense(dim, activation='relu'))

        decoder = keras.Sequential()
        decoder.add(layers.Input(shape=(self.latent_dim,)))
        for dim in decoder_dims:
            decoder.add(layers.Dense(dim, activation='relu'))
        decoder.add(layers.Dense(input_dim, activation='sigmoid'))

        autoencoder = keras.Model(encoder.input, decoder(encoder.output))
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        self.model = autoencoder
        self.encoder = encoder
        self.decoder = decoder

        self.encode = encoder.predict
        self.decode = decoder.predict

class AE3:
    def __init__(
        self,
        input_dim,
        features_spec,
        encoding_dim
    ):
        self.input_dim = input_dim

        input = layers.Input((input_dim,))
        sparse_features = layers.Dense(
            features_spec[0],
            activation="relu",
            kernel_regularizer=keras.regularizers.l1_l2(features_spec[1], features_spec[2])
        )(input)
        x = layers.concatenate([input, sparse_features])
        x = layers.Dense(encoding_dim, activation="relu")(x)
        encoder = keras.Model(input, x)

        decoder = keras.Sequential([
            layers.Input((encoding_dim,)),
            layers.Dense(input_dim, activation="sigmoid")
        ])

        autoencoder = keras.Model(input, decoder(encoder.output))
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        self.model = autoencoder
        self.encoder = encoder
        self.decoder = decoder

        self.encode = encoder.predict
        self.decode = decoder.predict

class Data:
    class Fit:
        def  __init__(self, data, ae):
            self.ae = ae
            self.data = data        

        def cor(self):
            from numpy import corrcoef
            return corrcoef(
                self.data.test.flatten(),
                self.decoded.flatten()
            )[0,1]

        def all_cor(self):
            import sklearn.preprocessing as sklp
            t = sklp.StandardScaler().fit_transform(self.data.test.T).T
            d = sklp.StandardScaler().fit_transform(self.decoded.T).T
            c = (t * d).mean(axis=1)
            return c

        def cor_plot(
            self,
            n_samples = 50000,
            kernel = 'gaussian',
            bandwidth = 0.1,
            nbins = 50
        ):
            import sklearn.neighbors as skln
            import matplotlib.pyplot as plt

            x = self.data.test.flatten()
            y = self.decoded.flatten()
            i = np.random.randint(0, len(x), n_samples)
            x = x[i]
            y = y[i]

            k = skln.KernelDensity(
                kernel=kernel,
                bandwidth=bandwidth
            ).fit(np.vstack([x, y]).T)
            xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
            zi = np.vstack([xi.flatten(), yi.flatten()]).T
            zi = k.score_samples(zi).reshape(xi.shape)
            plt.pcolormesh(xi, yi, zi, cmap=plt.cm.terrain)
            plt.contour(xi, yi, zi)

        @property
        def train_encoded(self):
            for data in self.data.train.batch(1):
                mat = data[0].numpy()
                yield (mat, self.ae.encode(mat))

        @property
        def train_decoded(self):
            for mat, enc in self.train_encoded:
                yield (mat, self.ae.decode(enc))

        @property
        def test_encoded(self):
            for data in self.data.test.batch(1):
                mat = data[0].numpy()
                yield (mat, self.ae.encode(mat))

        @property
        def test_decoded(self):
            for mat, enc in self.test_encoded:
                yield (mat, self.ae.decode(enc))

    def fit(self, ae, **kwargs):
        return self.Fit(self, ae, **kwargs)


class Sparse(layers.Layer):
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

