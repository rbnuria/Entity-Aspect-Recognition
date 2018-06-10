import numpy as np
from csv_embeddings import *
from xml_data import *
import tensorflow as tf 
from sklearn.model_selection import train_test_split
import random
from keras.utils import to_categorical
from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import concatenate
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import TimeDistributed
from keras.layers import Permute
from keras.layers import Reshape
from keras.layers import maximum
from keras.layers import Lambda
#*************************************************************#
#						PREPROCESAMIENTO 					  #
#*************************************************************#

import warnings
warnings.filterwarnings("ignore")

#Obtenemos word_embeddings
embeddings_path = '../AmazonWE/sentic2vec.csv'
line_embeddings = CSVEmbeddings(embeddings_path).embeddings

#Fijamos la semilla para poder reproducir experimentos
random.seed(123456)

#tamaño de test
test_size = 0.3

#Tamaño de la ventana
windows_size = 3


def longerSentence(data):
	max_sentence_length = 0
	for sentence in data:
		max_sentence_length = max(max_sentence_length, len(sentence))

	return max_sentence_length


data_source = 'Laptop_Train_v2.xml'
data = XMLData(data_source).getIobData()


#Establecemos el tamaño máximo a 65
MAX_LENGTH = 65
print("Longest sentence: %d" % MAX_LENGTH)



def padding_truncate(training_sentences):
    """Amplia o recorta las oraciones de entrenamiento.
    
    Args:
        training_sentences: Lista de listas de enteros.

    -> En este caso solo ampliaría pues estamos tomando como tamaño la frase más larga     
    """
    
    for i in range(len(training_sentences)):
        sent_size = len(training_sentences[i])
        if sent_size > MAX_LENGTH:
            training_sentences[i] = training_sentences[i][:MAX_LENGTH]
        elif sent_size < MAX_LENGTH:
            training_sentences[i] += [0] * (MAX_LENGTH - sent_size)
    
    return training_sentences 
            


def labelsToIdx(label):

	aux = 0

	if(label == 'O'):
		aux = 1
	elif(label == 'B'):
		aux = 2
	elif(label == 'I'):
		aux = 3

	return aux

def dataPreparation(sentences, labels, word_to_index):
	'''
		Método que prepara los datos -> padding a las frases al tamaño de la frase más larga

	'''

	x_matrix = []
	y_matrix = []

	padding_index = word_to_index["PADDING"]
	unkown_index = word_to_index["UNKOWN"]



	for sentence in sentences:

		wordIndices = []

		for word in sentence:
			if word in word_to_index:
				wordIndices.append(word_to_index[word])
			elif word.lower() in word_to_index:
				wordIndices.append(word_to_index[word.lower()])
			else:
				wordIndices.append(unkown_index)

		x_matrix.append(wordIndices)

	#Padding a todas las frases
	x_matrix = padding_truncate(x_matrix)


	for fila_labels in labels:
		labelIndices = []

		for label in fila_labels:
			labelIndices.append(labelsToIdx(label))

		y_matrix.append(labelIndices)

	#Padding a las etiquetas
	y_matrix = padding_truncate(y_matrix)

	return (x_matrix, y_matrix)

#*************************************************************#
#					PREPARACIÓN DATOS						  #
#*************************************************************#


# Conjunto de datos que vamos a utilizar

labels = []
sentences = []
words = {}

#Para cada frase del conjunto de datos
for sentence in data:
	fila_labels = []
	fila_sentences = []
	for word, label in sentence:
		fila_labels.append(label)
		fila_sentences.append(word)
		words[word] = True

	fila_labels = np.array(fila_labels)
	labels.append(fila_labels)
	sentences.append(fila_sentences)


# Análogamente para las palabras
word_to_index = {}
wordEmbeddings = []

for emb in line_embeddings:
	word = emb[0]

	#Si es la primera vez que entramos creamos los embeddings para palabras desconocidas y padding
	if len(word_to_index) == 0:
		word_to_index["PADDING"] = len(word_to_index)
		vector_aux = np.zeros(len(emb)-1)
		wordEmbeddings.append(vector_aux)


		word_to_index["UNKOWN"] = len(word_to_index)
		vector_aux = np.random.uniform(-0.25, 0.25, len(emb)-1)
		wordEmbeddings.append(vector_aux)


	#Si la palabra está en nuestro diccionario (caso concreto que estemos ejecutando) -> Lo añadimos a wordEmbeddings
	if word.lower() in words:
		vector_aux = np.array([float(num) for num in emb[1:]])
		wordEmbeddings.append(vector_aux)
		word_to_index[word] = len(word_to_index)


wordEmbeddings = np.array(wordEmbeddings)
embeddings = {'wordEmbeddings': wordEmbeddings, 'word2Idx': word_to_index}


##############################################################################################
# 									PARTICIONAMIENTO										 #
##############################################################################################

#Creamos las particiones de entrenamiento y test a partir de data.
#obtenemos los datos
tokens, labels = dataPreparation(sentences, labels, word_to_index)

tokens_train, tokens_test, labels_train, labels_test = train_test_split(tokens, labels, test_size=test_size, shuffle=False)

tokens_train = np.array(tokens_train)
tokens_test = np.array(tokens_test)
labels_train = np.array(labels_train)
labels_test = np.array(labels_test)


data = {
    'wordEmbeddings': wordEmbeddings, 'word2Idx': word_to_index,
    'train': {'sentences': tokens_train, 'labels': labels_train},
    'test':  {'sentences': tokens_test, 'labels': labels_test}
    }

word_embeddings = data['wordEmbeddings']


x_train = data['train']['sentences']
x_test = data['test']['sentences']

#Colocamos los embeddings a mano -> problema con las dimensiones si no
X_train = []
for sentence in x_train:

	frase_embeddings = []

	for word in sentence:
		emb_word = word_embeddings[word]
		frase_embeddings.append(emb_word)

	X_train.append(frase_embeddings)

X_train = np.asarray(X_train)

X_test = []
for sentence in x_test:

	frase_embeddings = []

	for word in sentence:
		emb_word = word_embeddings[word]
		frase_embeddings.append(emb_word)

	X_test.append(frase_embeddings)

X_test = np.asarray(X_test)



y_train = data['train']['labels']
y_test = data['test']['labels']


print('X_train shape:', x_train.shape)
print('X_test shape:', x_test.shape)
print('Y_train shape:', y_train.shape)
print('Y_test shape;', y_test.shape)



#*************************************************************#
#					CREACIÓN DE LA RED - CNN 				  #
#*************************************************************#

#Configuración parámetros
batch_size = 50

nb_filters = 100

#Poria utiliza [3,2]
filter_lengths = [2,3]
hidden_dims = 100
nb_epoch = 1

#Tamaño de entrada -> longitud de la frase más larga
input_size = MAX_LENGTH

#DEFINIMOS EL MODELO
sequence_input = Input(shape = (input_size, ), dtype = 'float64')
embedding_layer = Embedding(word_embeddings.shape[0], word_embeddings.shape[1], weights=[word_embeddings],trainable=False, input_length = input_size) #Trainable false
embedded_sequence = embedding_layer(sequence_input)
#Primera convolución
x = Conv1D(filters = 100, kernel_size = 2, activation = "tanh")(embedded_sequence)
x = MaxPooling1D(pool_size = 2)(x)
x = Dropout(0.5)(x)
#Segunda
x = Conv1D(filters = 50, kernel_size = 3, activation = "tanh")(x)
x = MaxPooling1D(pool_size = 2)(x)
x = Dropout(0.5)(x)
#Transponemos -> Dense -> transponemos
x = Permute((2,1))(x)
x = Dense(65, activation = "relu")(x)
x = Permute((2, 1))(x)

#Una probabilidad por etiqueta
preds = TimeDistributed(Dense(4, activation="softmax"))(x)

model = Model(sequence_input, preds)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

#Pasamos las etiquetas a categóricas 
y_train_categorical = []
for label in y_train:
	y_train_categorical.append(to_categorical(label,4))


#Pasamos lista -> numpy array
y_train_categorical = np.array(y_train_categorical)

model.fit(x = x_train, y = y_train_categorical, batch_size = 128, epochs = 10)

       


