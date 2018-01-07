from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard

import sys
import scipy.io
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
import numpy as np

def simple_dnn_model(encoding_dim):
    input_img = Input(shape=(784,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)
    # autoencoder
    autoencoder = Model(input_img, decoded)
    # encoder
    encoder = Model(input_img, encoded)
    # decoder
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    decoder.summary()

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder, encoder, decoder

def deep_dnn_model(encoding_dim):
    input_img = Input(shape=(784,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)

    # autoencoder
    autoencoder = Model(input_img, decoded)
    # encoder
    encoder = Model(input_img, encoded)
    # decoder
    #encoded_input = Input(shape=(encoding_dim,))
    #decoded = autoencoder.layers[-3](encoded_input)
    #decoder = Model(encoded_input, decoded_output)
    #decoder.summary()
    decoder =  encoder  
    
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder, encoder, decoder

def autoencoder(model_type, path, encoding_dim):
    f_mnist = scipy.io.loadmat(path)
    
    x_train, y_train = f_mnist['X_train'], f_mnist['y_train']
    x_test, y_test = f_mnist['X_test'], f_mnist['y_test']

    x_train, x_test = x_train.astype('float32') / 255. , x_test.astype('float32') / 255.
    x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784)

    print('x_train:',x_train.shape, 'x_test:', x_test.shape)
    print('y_train:',y_train.shape, 'y_test:', y_test.shape)

    if model_type == '0':
        autoencoder, encoder, decoder = simple_dnn_model(encoding_dim)
        save_path = 'simple_dnn.png'
        model_name = 'simple_dnn_model.h5'
    elif model_type == '1':
        autoencoder, encoder, decoder = deep_dnn_model(encoding_dim)
        save_path = 'deep_dnn.png'
        model_name = 'deep_dnn_model.h5'

    earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')
    autoencoder.fit(x_train, x_train, epochs=100, batch_size=256, 
                    callbacks=[earlystopping, TensorBoard(log_dir='./board/')],
                    shuffle=True, validation_data=(x_test, x_test))

    autoencoder.save(model_name)
    print('Model is saved')

    #encoder_img = encoder.predict(x_test)
    #decoder_img = decoder.predict(encoder_img)
    img = autoencoder.predict(x_test)

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(img[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(save_path)
    '''
    n = 4  # how many digits we will display
    #plt.figure(figsize=(20, 4))
    for i in range(n):
        plt.imsave('./pic/'+str(i)+'.png',x_test[i].reshape(28, 28), cmap='Greys_r')
        plt.imsave('./pic/'+str(i)+'_.png',img[i].reshape(28, 28), cmap='Greys_r')
    '''
    return

if __name__ == '__main__':
    autoencoder(model_type=sys.argv[1], path='./fashion_mnist_dataset.mat', encoding_dim=32)
