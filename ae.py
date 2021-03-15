import sklearn.decomposition as skld
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow as tf

class AE:
    def __init__(
        self,
        input_dim, encoding_dim,
        encoder_activation,
        decoder_activation,
        optim,
        loss
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
        
class PCA:
    def __init__(
        self,
        encoding_dim
    ):
        self.encoding_dim = encoding_dim

        self.model = skld.PCA(n_components=encoding_dim)

        self.encode = self.model.transform
        self.decode = self.model.inverse_transform

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

class AE2:
    def __init__(
        self,
        input_shape
    ):
        self.input_shape = input_shape

        encoder_input = keras.Input(shape=(np.prod(input_shape),))
        x = layers.Reshape(target_shape=input_shape)(encoder_input)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        encoder = keras.Model(encoder_input, x)

        decoder_input = keras.Input(shape=encoder.output.shape[1:].as_list())
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(decoder_input)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation='relu')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        x = layers.Reshape(target_shape=(np.prod(input_shape),))(x)
        decoder = keras.Model(decoder_input, x)

        autoencoder_input = keras.Input(shape=(np.prod(input_shape),))
        autoencoder = keras.Model(autoencoder_input, decoder(encoder(autoencoder_input)))

        autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1, epsilon=1e-14), loss='binary_crossentropy')

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
    
class MNIST(Data):
    def __init__(self):
        from keras.datasets import mnist
        import numpy as np
        (x_train, _), (x_test, _) = mnist.load_data()

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        self.train = x_train
        self.test = x_test

    class Fit(Data.Fit):
        def plot(self):
            import matplotlib.pyplot as plt

            n = 10  # How many digits we will display
            plt.figure(figsize=(20, 4))
            for i in range(n):
                # Display original
                ax = plt.subplot(2, n, i + 1)
                plt.imshow(self.data.test[i].reshape(28, 28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # Display reconstruction
                ax = plt.subplot(2, n, i + 1 + n)
                plt.imshow(self.decoded[i].reshape(28, 28))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()

