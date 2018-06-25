from model import *
import numpy as np
from keras.utils import to_categorical
from keras.layers import Embedding
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import TimeDistributed
from keras.models import Model
from keras.layers import LSTM
from keras import regularizers


class LSTM_Model(Model_RRNN):
	def __init__(self,embeddings_path, data_path, max_length, batch_size, test_size):
		super(LSTM_Model, self).__init__(embeddings_path, data_path, max_length, batch_size, test_size)
		
		self.model = None

	def trainModel(self):
		input_size = self.max_length
		sequence_input = Input(shape = (input_size, ), dtype = 'float64')
		embedding_layer = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1], weights=[self.word_embeddings],trainable=False, input_length = input_size) #Trainable false
		embedded_sequence = embedding_layer(sequence_input)
		#Primera convolución
		x = LSTM(units = 128, return_sequences=True)(embedded_sequence)
		x = Dropout(0.5)(x)
		x = Dense(100, activity_regularizer=regularizers.l1(0.05))(x)
		x = Dropout(0.5)(x)

		#Una probabilidad por etiqueta
		preds = TimeDistributed(Dense(4, activation="softmax"))(x)

		self.model = Model(sequence_input, preds)

		self.model.summary()

		self.model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

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

