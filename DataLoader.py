import numpy as np

import cv2
import tensorflow as tf


class DataLoader(tf.keras.utils.Sequence):
    
    def __init__(self, list_IDs, labels, batch_size=32, dim=50, n_channels=1, shuffle=True, n_classes=6, train_dir=''):
        
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.train_dir = train_dir
        self.on_epoch_end()

    def preprocess(self, img):
	    image = cv2.imread(img)
	    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	    res = cv2.resize(gray, (self.dim,self.dim))
	    return (res/255).reshape(-1,self.dim,self.dim,1)
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim, self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.preprocess(self.train_dir +'/'+ ID)
            # Store class
            y[i] = self.labels[ID]

        return X, y
    

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
