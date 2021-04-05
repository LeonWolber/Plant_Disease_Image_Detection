import tensorflow as tf

from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from DataLoader import *



class PlantModel(DataLoader):

	def __init__(self, dim, n_channels, optimizer = 'adam', activation = 'relu', learning_rate_ = 0.001, n_classes=6):
		self.dim = dim
		self.n_channels = n_channels
		self.learning_rate_ = learning_rate_
		if optimizer == 'adam':
			self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_)
		else:
			self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_, momentum=0.9, nesterov=True)
		self.activation = activation
		self.n_classes = n_classes


	def build_model(self):
		model = Sequential()
		model.add(Conv2D(32, kernel_size=(3, 3), activation=self.activation, input_shape=(self.dim,self.dim, self.n_channels),padding='same'))
		#model.add(LeakyReLU(alpha=0.1))
		model.add(MaxPooling2D((2, 2), padding='same'))

		model.add(Conv2D(64, (3, 3), activation=self.activation,padding='same'))
		#model.add(LeakyReLU(alpha=0.1))
		model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

		model.add(Conv2D(128, (3, 3), activation=self.activation,padding='same'))
		#model.add(LeakyReLU(alpha=0.1))                  
		model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

		model.add(Flatten())

		model.add(Dense(128, activation=self.activation))
		#model.add(LeakyReLU(alpha=0.1))                  
		model.add(Dense(self.n_classes, activation='softmax'))


		model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics='accuracy')
		return model

