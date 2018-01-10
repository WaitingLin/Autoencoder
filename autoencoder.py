from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard

import sys
import scipy.io
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
import numpy as np

def simple_dnn_model():
    input_img = Input(shape=(784,))
    encoded = Dense(32, activation='relu')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)
    # autoencoder
    autoencoder = Model(input_img, decoded)
    # encoder
    encoder = Model(input_img, encoded)
    # decoder
    encoded_input = Input(shape=(32,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    decoder.summary()

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder, encoder, decoder

def deep_dnn_model():
    input_img = Input(shape=(784,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)

    # autoencoder
    autoencoder = Model(input_img, decoded)
    # encoder
    encoder = Model(input_img, encoded)
    # decoder
    encoded_input = Input(shape=(32,))
    deco = autoencoder.layers[-3](encoded_input)
    deco = autoencoder.layers[-2](deco)
    deco = autoencoder.layers[-1](deco)
    decoder = Model(encoded_input, deco)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder, encoder, decoder

def convolution_model():
    input_img = Input(shape=(28,28,1))    
    
    encoded = Conv2D(16, (5,5), activation='relu', padding='same')(input_img)
    encoded = MaxPooling2D((2,2), padding='same')(encoded)
    encoded = Conv2D(8, (5,5), activation='relu', padding='same')(encoded)
    encoded = MaxPooling2D((2,2), padding='same')(encoded)

    decoded = Conv2D(8, (5,5), activation='relu', padding='same')(encoded)
    decoded = UpSampling2D((2,2))(decoded)
    decoded = Conv2D(16, (5,5), activation='relu', padding='same')(decoded)
    decoded = UpSampling2D((2,2))(decoded)
    decoded = Conv2D(1, (5,5), activation='sigmoid', padding='same')(decoded)

    # autoenocder
    autoencoder = Model(input_img, decoded)
    # enocder
    encoder = Model(input_img, encoded)
    # decoder
    encoded_input = Input(shape=(7,7,8))
    deco = autoencoder.layers[-5](encoded_input)
    deco = autoencoder.layers[-4](deco)
    deco = autoencoder.layers[-3](deco)
    deco = autoencoder.layers[-2](deco)
    deco = autoencoder.layers[-1](deco)
    decoder = Model(encoded_input, deco)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder, encoder, decoder


def autoencoder(model_type, path):
    f_mnist = scipy.io.loadmat(path)
    
    x_train, y_train = f_mnist['X_train'], f_mnist['y_train']
    x_test, y_test = f_mnist['X_test'], f_mnist['y_test']

    x_train, x_test = x_train.astype('float32') / 255. , x_test.astype('float32') / 255.

    print('x_train:',x_train.shape, 'x_test:', x_test.shape)
    print('y_train:',y_train.shape, 'y_test:', y_test.shape)

    if model_type == '0':
        autoencoder, encoder, decoder = simple_dnn_model()
        save_path = 'simple_dnn.png'
        model_name = 'simple_dnn_model.h5'
        log_dir = './simple_dnn_board'
        x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784)
    elif model_type == '1':
        autoencoder, encoder, decoder = deep_dnn_model()
        save_path = 'deep_dnn.png'
        model_name = 'deep_dnn_model.h5'
        log_dir = './deep_dnn_board'
        x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784)
        encoder.save('deep_enocder.h5')
    elif model_type == '2':
        autoencoder, encoder, decoder = convolution_model()
        save_path = 'convolutional.png'
        model_name = 'convolutional_model.h5'
        log_dir = './convolutional_board'
        x_train, x_test = np.reshape(x_train, (len(x_train), 28, 28, 1)), np.reshape(x_test, (len(x_test), 28, 28, 1))
    else:
        print(' = =? ')
    #earlystopping = EarlyStopping(monitor='loss', patience=1, verbose=1, mode='min')
    autoencoder.summary()
    autoencoder.fit(x_train, x_train, epochs=500, batch_size=128, 
                    callbacks=[TensorBoard(log_dir=log_dir)],
                    shuffle=True, validation_data=(x_test, x_test))

    autoencoder.save(model_name)
    print('Model is saved')

    #encoder_img = encoder.predict(x_test)
    #decoder_img = decoder.predict(encoder_img)
    img = autoencoder.predict(x_test)

    # plot
    n = 10 
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(img[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(save_path)
    return

if __name__ == '__main__':
    autoencoder(model_type=sys.argv[1], path='./fashion_mnist_dataset.mat')
