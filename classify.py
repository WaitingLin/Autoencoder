from keras.models import load_model
from sklearn.cluster import KMeans

import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np

def main():
    f_mnist = scipy.io.loadmat('./fashion_mnist_dataset.mat')
    x_test = f_mnist['X_test']
    x_test_normalize =  x_test.astype('float32') / 255.
    
    model = load_model('./deep_encoder.h5')
    
    xm = model.predict(x_test_normalize)
    kmeans = KMeans(n_clusters=10).fit(xm)
    labels = kmeans.labels_

    #print(labels[0:10])
    n = 100
    plt.figure(figsize=(20, 4))
    for i in range(n):
        plt.imsave('./pic/'+str(labels[i])+'/'+str(i)+'.png', x_test[i].reshape(28, 28), cmap='Greys_r')

if __name__ == '__main__':
    main()
