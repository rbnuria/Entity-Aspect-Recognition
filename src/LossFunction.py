import numpy as np
import tensorflow as tf
from keras import backend as K

class Lossfunction:

	def __init__(self, batch_size):
		self.transition_params = None
		self.batch_size = batch_size

	def loss(self, y_true, y_pred):
		array_lengths = np.repeat(65, self.batch_size)
		sequence_lengths = K.variable(array_lengths, name = "sequence_lengths")
		

		y_true = tf.argmax(y_true, axis = -1)
		y_true = tf.cast(y_true, tf.int32)
		log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(y_pred, y_true, sequence_lengths = array_lengths, transition_params=self.transition_params)

		#Actualización matriz de pesoss
		self.transition_params = transition_params
		loss = tf.reduce_mean(-log_likelihood)

		return loss