import numpy as np
import matplotlib.pyplot as plt
import keras.layers
import keras.optimizers
import keras.models

class DFModel:
    def __init__():
        self.model = self._configure_model()

    def _configure_model(self):
        # Input layer
        # Height, Width, # of channels
        X = Input(shape = (720, 1280, 3))

        # Each batch below - is convolutional block
        
        # It is a convolutional layer
        # First parameter is a dimensionality of output space
        # Second parameter is a kernel size - size of convolutional window
        # Parameter 'padding' sets whether the dimensionalty of input is the same as of output
        # Parameter 'activation' sets the type of activation function
        # Double parentheses syntax - is specific of Keras model initialization
        X_1 = Conv2D(8, (3,3), padding='same', activation='relu')(x)
        # This layer does a data normalization:
        # Transform data in such a way that mean is near 0 and sd is near 1
        X_1 = BatchNormalization()(X_1)
        # This layer reduces dimensionality of output by taking the max value out of each "pool"
        # Parameter pool_size define the size of pool
        X_1 = MaxPooling2D(pool_size=(2,2), padding='same')(X_1)

        # The same explanation as below is applicable to each code batch below
        X_2 = Conv2D(8, (5,5), padding='same', activation='relu')(X_1)
        X_2 = BatchNormalization()(X_2)
        X_2 = MaxPooling2D(pool_size=(2,2), padding='same')(X_2)


        X_3 = Conv2D(16, (5,5), padding='same', activation='relu')(X_2)
        X_3 = BatchNormalization()(X_3)
        X_3 = MaxPooling2D(pool_size=(2,2), padding='same')(X_3)

        X_4 = Conv2D(16, (5,5), padding='same', activation='relu')(X_3)
        X_4 = BatchNormalization()(X_4)
        X_4 = MaxPooling2D(pool_size=(2,2), padding='same')(X_4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLu(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return Model(inputs=x, outputs=y)