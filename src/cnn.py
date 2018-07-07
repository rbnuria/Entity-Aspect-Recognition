import numpy as np
from cnn_model import *
import tensorflow as tf 
import random


#Obtenemos word_embeddings
#embeddings_path = '../deps.words.bz2'#
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
batch_size = 16


#*************************************************************#
#					CREACIÓN DE LA RED - CNN 				  #
#*************************************************************#

cnn_model = CNN_Model(embeddings_path, data_path, MAX_LENGTH, batch_size, test_size)

cnn_model.trainModel()
cnn_model.fitModel(epochs = 50)

pred = cnn_model.predictModel()

print(pred[0])
print((cnn_model.getLabelsTest())[0])

print("******************")

print(pred[1])
print((cnn_model.getLabelsTest())[1])

print("******************")

print(pred[2])
print((cnn_model.getLabelsTest())[2])


print("******************")

print(pred[3])
print((cnn_model.getLabelsTest())[3])


print("******************")

print(pred[4])
print((cnn_model.getLabelsTest())[4])
