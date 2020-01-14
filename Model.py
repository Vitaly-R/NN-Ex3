from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Activation, Reshape


class AEModel(Model):

    def __init__(self):
        super(AEModel, self).__init__()
        # activations
        self.relu = Activation('relu')
        self.sigmoid = Activation('sigmoid')
        # Encoder
        self.conv1 = Conv2D(32, 3, 2, 'valid')
        self.conv2 = Conv2D(64, 3, 2, 'valid')
        self.flatten = Flatten()
        self.dense1 = Dense(512)
        self.dense2 = Dense(10)
        # Decoder
        self.dense3 = Dense(512)
        self.dense4 = Dense(7 * 7 * 64)
        self.reshape = Reshape((7, 7, 64))
        self.convt1 = Conv2DTranspose(32, 2, 2, padding='valid')
        self.convt2 = Conv2DTranspose(1, 2, 2, padding='valid')

    def __call__(self, x, *args, **kwargs):
        y = self.relu(self.conv1(x))
        y = self.relu(self.conv2(y))
        y = self.flatten(y)
        y = self.relu(self.dense1(y))
        y = self.relu(self.dense2(y))
        y = self.relu(self.dense3(y))
        y = self.relu(self.dense4(y))
        y = self.reshape(y)
        y = self.relu(self.convt1(y))
        y = self.convt2(y)
        return self.sigmoid(y)

    def encode(self, x):
        y = self.relu(self.conv1(x))
        y = self.relu(self.conv2(y))
        y = self.flatten(y)
        y = self.relu(self.dense1(y))
        y = self.relu(self.dense2(y))
        return y
