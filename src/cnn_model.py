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

	def __init__(self,embeddings = None, vocabulary = None, train_path = None, test_path = None, max_length = None, batch_size = None, filtro_1 = None, filtro_2 = None, kernel_1 = None, kernel_2 = None, dropout_1 = None, dropout_2 = None):
		super(CNN_Model, self).__init__(embeddings, vocabulary, train_path, test_path, max_length, batch_size)
		
		self.filter_1 = filtro_1
		self.filter_2 = filtro_2
		self.kernel_1 = kernel_1
		self.kernel_2 = kernel_2
		self.dropout_1 = dropout_1
		self.dropout_2 = dropout_2

		self.model = None

		self.configureModel()


	def configureModel(self):
		'''
			Creación de la arquitectura
		'''
		input_size = self.max_length
		
		#Inputs y embeddings
		sequence_input = Input(shape = (input_size, ), dtype = 'float64', name = "input_data")
		sequence_input_lengths = Input(shape = (1, ), dtype = 'int32', name="input_sequence_lengths")
		#Definimos función de pérdida -> en función de sequence_input_lengths
		self.loss_object = Lossfunction(self.batch_size, sequence_input_lengths)

		embedding_layer = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1], weights=[self.word_embeddings],trainable=False, input_length = input_size, name = "embeddings") #Trainable false
		embedded_sequence = embedding_layer(sequence_input)
		
		#Primera convolución
		x = Conv1D(filters = self.filter_1, kernel_size = self.kernel_1, padding="same", activation = "tanh", name ="first_convolution")(embedded_sequence)
		x = MaxPooling1D(pool_size = 2, strides=1, padding="same", name = "first_max_pooling")(x)
		x = Dropout(self.dropout_1, name = "first_dropout")(x)
		
		#Segunda
		x = Conv1D(filters = self.filter_2, kernel_size = self.kernel_2, padding="same", activation = "tanh", name = "second_convolution")(x)
		x = MaxPooling1D(pool_size = 2, strides=1, padding="same", name = "second_max_pooling")(x)
		x = Dropout(self.dropout_2, name = "second_dropout")(x)
		
		#Última capa
		preds = TimeDistributed(Dense(4, name = "last_layer"))(x)
		#preds = Dense(4, activation = "tanh", name = "last_layer")(x)
		#preds = Dense(4, name = "last_layer")(x)

		#Creamos el modelo
		self.model = Model(input = [sequence_input, sequence_input_lengths], output = [preds])

		self.model.summary()

	def trainModel(self):
		'''
			Definición y compilación del modelo.
		'''
		

		self.model.compile(loss=self.loss_object.loss, optimizer = 'adam', metrics=['accuracy'])

		#Build
		self.model.build(None)

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


		self.model.fit(x = {"input_data":self.x_train, "input_sequence_lengths": np.array(self.sequence_lengths_train)}, y = y_train_categorical, batch_size = self.batch_size, epochs = epochs, verbose = 1)

	def predictModel(self, transition_params = None):
		'''
			Predicción de las etiquetas de test.
			Utilizamos algoritmo viterbi para obtener la mejor secuencia.
		'''


		logits = self.model.predict([self.x_test, np.array(self.sequence_lengths_test)])

		if transition_params == None:
			trans_params = K.eval(self.loss_object.getTransitionParams())
		else:
			trans_params = K.eval(transition_params)

		viterbi_sequences = []

		for logit, sequence_length in zip(logits, np.array(self.sequence_lengths_test)):
			logit = logit[:sequence_length]
			viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
			viterbi_sequences += [viterbi_seq]

		self.predicted_labels = np.array(self.padding_truncate(viterbi_sequences))

		return self.predicted_labels

