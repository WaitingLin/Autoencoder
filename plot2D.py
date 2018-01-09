from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def autoencoder(path):
    f_mnist = scipy.io.loadmat(path)

    x_train, y_train = f_mnist['X_train'], f_mnist['y_train']
    x_test, y_test = f_mnist['X_test'], f_mnist['y_test']
    
    y_test = y_test.reshape(-1,)

    #x_train, x_test = x_train.astype('float32') / 255. , x_test.astype('float32') / 255.

    input_img = Input(shape=(784,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    #encoded = Dense(2, activation='relu', kernel_regularizer=regularizers.l2(0.01))(encoded)
    encoded = Dense(2, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='relu')(decoded)
    
    # autoencoder
    autoencoder = Model(input_img, decoded)
    # encoder    
    encoder = Model(input_img, encoded)
    # decoder
    encoder_input = Input(shape=(2,))
    deco = autoencoder.layers[-3](encoder_input)
    deco = autoencoder.layers[-2](deco)
    deco = autoencoder.layers[-1](deco)
    decoder = Model(encoder_input, deco)

    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(x_train, x_train, epochs=100, batch_size=512, shuffle=True, validation_data=(x_test, x_test))
    
    code = encoder.predict(x_test)
    
    plt.figure(figsize=(20,20))
    cm = plt.cm.get_cmap('RdYlBu')
    plt.scatter(code[:, 0], code[:, 1], c=y_test, cmap=cm)
    plt.colorbar()
    plt.savefig('2D.png')


    n = 20
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = np.linspace(0, 1000, n)
    grid_y = np.linspace(1000, 2000, n)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample) 
            x_decoded = x_decoded * 255.
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[(n-i-1) * digit_size: (n-i) * digit_size,  j * digit_size: (j+1) * digit_size] = digit
    plt.gray()
    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.savefig('reconstruction.png', bbox_inches='tight')
    return

if __name__ == '__main__':
    autoencoder('./fashion_mnist_dataset.mat')
