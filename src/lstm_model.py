from model import *
import numpy as np
from keras.utils import to_categorical
from LossFunction import * 
from keras.layers import Embedding, Bidirectional
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import TimeDistributed
from keras.models import Model
from keras.layers import LSTM
from keras import regularizers
import tensorflow as tf
import keras.backend as K

class LSTM_Model(Model_RRNN):
	def __init__(self,embeddings_path, train_path, test_path, max_length, batch_size, test_size):
		super(LSTM_Model, self).__init__(embeddings_path, train_path, test_path, max_length, batch_size, test_size)
		
		#Iniciamos session
		#K.set_session(tf.Session())

		self.model = None	

	def trainModel(self):
		'''
			Definición y compilación del modelo.
		'''
		input_size = self.max_length
		sequence_input = Input(shape = (input_size, ), dtype = 'float64', name  = "input_data")
		sequence_input_lengths = Input(shape = (1, ), dtype = 'int32', name="input_sequence_lengths")

		#Definimos función de pérdida -> en función de sequence_input_lengths
		self.loss_object = Lossfunction(self.batch_size, sequence_input_lengths)

		embedding_layer = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1], weights=[self.word_embeddings],trainable=False, input_length = input_size) #Trainable false
		embedded_sequence = embedding_layer(sequence_input)
		
		#LSTM
		x = Bidirectional(LSTM(units = 128, return_sequences=True))(embedded_sequence)
		#x = Dropout(0.25)(x)
		#x = Dense(128, activation = "tanh", activity_regularizer=regularizers.l1(0.001))(x)
		#x = Dropout(0.25)(x)
		#x = Dense(64,  activation = "tanh", activity_regularizer=regularizers.l2(0.001))(x)
		#x = Dropout(0.5)(x)
		#x = Dense(32,  activation = "tanh", activity_regularizer=regularizers.l2(0.01))(x)
		#x = Dropout(0.5)(x)

		#Una probabilidad por etiqueta
		#preds = TimeDistributed(Dense(4, activation="softmax"))(x)
		#preds = Dense(4, name = "last_layer")(x)
		preds = TimeDistributed(Dense(4, name = "last_layer"))(x)


		self.model = Model(input = [sequence_input, sequence_input_lengths], output = [preds])

		self.model.summary()

		self.model.compile(loss=self.loss_object.loss, optimizer = 'adam', metrics=['accuracy'])

	def fitModel(self, epochs):
		'''
			Ajuste del modelo con los datos del train.
		'''
		#Pasamos las etiquetas a categóricas 
		y_train_categorical = []
		for label in self.y_train:
			y_train_categorical.append(to_categorical(label,4))

		#Pasamos lista -> numpy array
		y_train_categorical = np.array(y_train_categorical)

		self.model.fit(x = {"input_data":self.x_train, "input_sequence_lengths": np.array(self.sequence_lengths_train)}, y = y_train_categorical, batch_size = self.batch_size, epochs = epochs, verbose = 1)

	def predictModel(self):
		'''
			Predicción de las etiquetas de test.
		'''
		logits = self.model.predict([self.x_test, np.array(self.sequence_lengths_test)])
		trans_params = K.eval(self.loss_object.getTransitionParams())

		viterbi_sequences = []

		#array_lengths = np.repeat(65, self.batch_size)
		#sequence_lengths = K.constant(array_lengths, name = "sequence_lengths")

		for logit, sequence_length in zip(logits, np.array(self.sequence_lengths_test)):
			logit = logit[:sequence_length]
			viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
			viterbi_sequences += [viterbi_seq]

		self.predicted_labels = np.array(self.padding_truncate(viterbi_sequences))

		return self.predicted_labels

	


