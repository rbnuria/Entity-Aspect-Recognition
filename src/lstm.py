import numpy as np
from lstm_model import *
import tensorflow as tf 
import random

#Obtenemos word_embeddings
embeddings_path = '../AmazonWE/sentic2vec.csv'
#Fijamos la semilla para poder reproducir experimentos
random.seed(123456)
#tamaño de test
test_size = 0.3
#Path de los datos
data_path = 'Laptop_Train_v2.xml'
#Establecemos el tamaño máximo a 65
MAX_LENGTH = 65
#Definimos batch_size
batch_size = 128


#*************************************************************#
#					CREACIÓN DE LA RED - LSTM 				  #
#*************************************************************#

lstm_model = LSTM_Model(embeddings_path, data_path, MAX_LENGTH, batch_size, test_size)

lstm_model.trainModel(1)
lstm_model.fitModel(epochs = 10)
