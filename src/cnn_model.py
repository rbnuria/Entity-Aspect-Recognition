from model import *
import numpy as np
from LossFunction import * 
from keras.utils import to_categorical
from keras.layers import Embedding
from keras.layers import Input
from keras.layers.core import Dropout, Flatten
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
		
		#self.loss_object = Lossfunction(self.batch_size, self.sequence_lengths)
		self.model = None

	def trainModel(self):
		'''
			Definición y compilación del modelo.
		'''
		input_size = self.max_length
		sequence_input = Input(shape = (input_size, ), dtype = 'float64', name="input_x_data")
		sequence_input_lengths = Input(shape=(1,), dtype="int32", name="input_x_sequence_lengths")
		self.loss_object = Lossfunction(self.batch_size, sequence_input_lengths)
		embedding_layer = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1], weights=[self.word_embeddings],trainable=False, input_length = input_size) #Trainable false
		embedded_sequence = embedding_layer(sequence_input)
		#Primera convolución
		x = Conv1D(filters = 100, kernel_size = 2, padding="same", activation = "tanh")(embedded_sequence)
		x = MaxPooling1D(pool_size = 2, strides=1, padding="same")(x)
		x = Dropout(0.5)(x)
		#Segunda
		x = Conv1D(filters = 50, kernel_size = 3, padding="same", activation = "tanh")(x)
		x = MaxPooling1D(pool_size = 2, strides=1, padding="same")(x)
		x = Dropout(0.5)(x)
		#Transponemos -> Dense -> transponemos
		#x = Permute((2,1))(x)
		preds = Dense(4, activation = "tanh")(x)
		#x = Permute((2, 1))(x)

		#Una probabilidad por etiqueta
		#preds = TimeDistributed(Dense(4, activation="softmax"), dtype = "int32")(x)
		self.model = Model(inputs=[sequence_input, sequence_input_lengths], outputs=[preds])

		self.model.summary()

		self.model.compile(loss=self.loss_object.loss, optimizer = 'adam', metrics=['accuracy'])

	def fitModel(self, epochs):
		'''
			Ajuste del modelo con los datos del train.
			TO DO: ajustar para datos de evaluación.
		'''

		#Pasamos las etiquetas a categóricas 
		y_train_categorical = []
		for label in self.y_train:
			y_train_categorical.append(to_categorical(label,4))

		#Pasamos lista -> numpy array
		y_train_categorical = np.array(y_train_categorical)

		self.model.fit(x = {"input_x_data":self.x_train, "input_x_sequence_lengths":np.array(self.sequence_lengths_train)}, y = y_train_categorical, batch_size = self.batch_size, epochs = epochs)

	def predictModel(self):
		'''
			Predicción de las etiquetas de test.
			Utilizamos algoritmo viterbi para obtener la mejor secuencia.
		'''
		logits = self.model.predict([self.x_test,np.array(self.sequence_lengths_test)])
		trans_params = K.eval(self.loss_object.getTransitionParams())

		viterbi_sequences = []

		array_lengths = np.repeat(65, self.batch_size)
		sequence_lengths = K.constant(array_lengths, name = "sequence_lengths")

		for logit, sequence_length in zip(logits, array_lengths):
			logit = logit[:sequence_length]
			viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
			viterbi_sequences += [viterbi_seq]

		return viterbi_sequences


