import numpy as np
from lstm_model import *
import tensorflow as tf 
import random

#Obtenemos word_embeddings
embeddings_path = '../glove.840B.300d.word2vec.txt'
#Fijamos la semilla para poder reproducir experimentos
random.seed(123456)
#tamaño de test
test_size = 0.3
#Path de los datos
train_path = 'data/Laptop_Train_v2.xml'
test_path = 'data/ABSA_Gold_TestData/Laptops_Test_Gold.xml'
#Establecemos el tamaño máximo a 65
MAX_LENGTH = 65
#Definimos batch_size
batch_size = 128


#*************************************************************#
#					CREACIÓN DE LA RED - LSTM 				  #
#*************************************************************#

lstm_model = LSTM_Model(embeddings_path, train_path, test_path, MAX_LENGTH, batch_size, test_size)

lstm_model.trainModel()
lstm_model.fitModel(epochs = 30)

pred = lstm_model.predictModel()

lstm_model.calculateAccuracy()

for i in range(0,10):
	print("******** " + str(i) + " *********")
	print(pred[i])
	print((lstm_model.getLabelsTest())[i])

