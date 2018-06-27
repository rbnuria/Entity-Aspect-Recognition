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
import tensorflow as tf
import keras.backend as K

class LSTM_Model(Model_RRNN):
	def __init__(self,embeddings_path, data_path, max_length, batch_size, test_size):
		super(LSTM_Model, self).__init__(embeddings_path, data_path, max_length, batch_size, test_size)
		
		self.model = None


	def custom_loss(self,y_pred, y_true):
		'''
			Método donde definimos función de pérdida.
			Queremos contabilizar el tanto por ciento de entidades (B / B{I}*) encontradas, en vez
			de las etiquetas individuales acertadas.
		'''

		labels_predicted = K.argmax(y_pred, axis = -1)
		labels = K.argmax(y_true, axis = -1)

		print("********")
		print(type(tf.Session().run(labels_predicted))) 
		print("**-----*")

		n_correct = 0
		count = 0
		idx = 0
		precision = 0
		batch = 1

		while batch < self.batch_size:
			print(batch)
			while idx < self.max_length:
				print(K.shape(labels_predicted[batch][idx]))
				if K.equal(labels_predicted[batch][idx],2) is not None: #He encontrado entidad
					
					count += 1

					if K.equal(labels_predicted[batch][idx], labels[batch][idx]) is not None:
						idx += 1
						found = True
						while idx < self.max_length and K.equal(labels_predicted[batch][idx],3) is not None: #Mientras sigo dentro de la misma entidad
							if K.equal(labels_predicted[batch][idx], labels[batch][idx]) is  None:
								found = False
								print("found false")

							idx += 1

						if idx < self.max_length:
							if K.equal(labels[batch][idx],3) is not None: #Si la entidad tenía más tokens de los predichos
								found = False
								print("found false")

						if found: #Sumamos 1 al número de encontrados
							n_correct += 1
							print("found true")

					else:
						print(idx)
						idx += 1

				else:
					idx += 1

			
			if count > 0:
				precision_aux = float(n_correct) / count

			precision = (precision + precision_aux)
			batch +=1
			idx = 0
			precision_aux = 0
			n_correct = 0
			count = 0

		return precision/self.batch_size

	def trainModel(self, loss_type):
		'''
			Definición y compilación del modelo.
		'''

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

		if(loss_type == 0):
			self.model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
		else:
			self.model.compile(loss=self.custom_loss, optimizer = 'adam', metrics=['accuracy'])

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

		self.model.fit(x = self.x_train, y = y_train_categorical, batch_size = self.batch_size, epochs = epochs)

	def predictModel(self):
		'''
			Predicción de las etiquetas de test.
		'''
		y_pred = self.model.predict(self.x_test)
		return y_pred

	


