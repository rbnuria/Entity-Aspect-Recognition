import numpy as np
from cnn_model import *
import tensorflow as tf 
import random


#Obtenemos word_embeddings
#embeddings_path = '../deps.words.bz2'#
#embeddings_path = '../glove.840B.300d.txt'
embeddings_path = '../glove.840B.300d.word2vec.txt'
#embeddings_path = '../glove.twitter.27B.200d.txt'

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
batch_size = 16


#*************************************************************#
#					CREACIÓN DE LA RED - CNN 				  #
#*************************************************************#

cnn_model = CNN_Model(embeddings_path, train_path, test_path, MAX_LENGTH, batch_size, test_size)

cnn_model.trainModel()
cnn_model.fitModel(epochs = 30)

pred = cnn_model.predictModel()

cnn_model.calculateAccuracy()

'''
print(pred[0])
print((cnn_model.getLabelsTrain())[0])

print("******************")

print(pred[1])
print((cnn_model.getLabelsTrain())[1])

print("******************")

print(pred[2])
print((cnn_model.getLabelsTrain())[2])


print("******************")

print(pred[3])
print((cnn_model.getLabelsTrain())[3])


print("******************")

print(pred[4])
print((cnn_model.getLabelsTrain())[4])
'''
