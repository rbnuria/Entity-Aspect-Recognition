import numpy as np
import tensorflow as tf
from keras import backend as K

class Lossfunction:

	def __init__(self, batch_size, sequence_lengths):
		self.transition_params = None
		self.batch_size = batch_size
		self.sequence_lengths = K.squeeze(sequence_lengths, axis=-1)

	def loss(self, y_true, y_pred):
		'''
			Método de definición de la función de pérdida en función del contexto.
			Introduce matriz de transición T 4x4 entre etiquetas.
			Basada en: https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html 
		'''

		y_true = tf.argmax(y_true, axis = -1)
		y_true = tf.cast(y_true, tf.int32)
		

		#sequence_lengths_batch = self.sequence_lengths[:self.batch_size]
		#self.sequence_lengths = self.sequence_lengths[self.batch_size:]
		print(self.sequence_lengths)
		
		#log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(y_pred, y_true, sequence_lengths = np.array(sequence_lengths_batch), transition_params=self.transition_params)
		log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(y_pred, y_true, sequence_lengths = self.sequence_lengths, transition_params=self.transition_params)

		#Actualización matriz de pesoss
		self.transition_params = transition_params

		#Crossentropy loss
		loss = tf.reduce_mean(-log_likelihood)

		return loss

	def getTransitionParams(self):
		return self.transition_params


	def getSequenceLength(self):
		return self.sequence_lengths