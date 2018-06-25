from model import *
import numpy as np
from LossFunction import * 
from keras.utils import to_categorical
from keras.layers import Embedding
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import TimeDistributed
from keras.layers import Permute
from keras.layers import Reshape
from keras.models import Model
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import GlobalMaxPooling1D

class CNN_Model(Model_RRNN):
	def __init__(self,embeddings_path, data_path, max_length, batch_size, test_size):
		super(CNN_Model, self).__init__(embeddings_path, data_path, max_length, batch_size, test_size)
		
		self.model = None

	def trainModel(self):
		loss_object = Lossfunction(self.batch_size)
		input_size = self.max_length
		sequence_input = Input(shape = (input_size, ), dtype = 'float64')
		embedding_layer = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1], weights=[self.word_embeddings],trainable=False, input_length = input_size) #Trainable false
		embedded_sequence = embedding_layer(sequence_input)
		#Primera convolución
		x = Conv1D(filters = 100, kernel_size = 2, activation = "tanh")(embedded_sequence)
		x = MaxPooling1D(pool_size = 2)(x)
		x = Dropout(0.5)(x)
		#Segunda
		x = Conv1D(filters = 50, kernel_size = 3, activation = "tanh")(x)
		x = MaxPooling1D(pool_size = 2)(x)
		x = Dropout(0.5)(x)
		#Transponemos -> Dense -> transponemos
		x = Permute((2,1))(x)
		x = Dense(65, activation = "relu")(x)
		x = Permute((2, 1))(x)

		#Una probabilidad por etiqueta
		preds = TimeDistributed(Dense(4, activation="softmax"), dtype = "int32")(x)

		self.model = Model(sequence_input, preds)

		self.model.summary()

		self.model.compile(loss=loss_object.loss, optimizer = 'adam', metrics=['accuracy'])

	def fitModel(self, epochs):
		#Pasamos las etiquetas a categóricas 
		y_train_categorical = []
		for label in self.y_train:
			y_train_categorical.append(to_categorical(label,4))

		#Pasamos lista -> numpy array
		y_train_categorical = np.array(y_train_categorical)

		self.model.fit(x = self.x_train, y = y_train_categorical, batch_size = self.batch_size, epochs = epochs)

	def predictModel(self):
		y_pred = self.model.predict(self.x_train)
		return y_pred

