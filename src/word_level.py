import numpy as np
from csv_embeddings import *
from xml_data import *
import tensorflow as tf 
from sklearn.model_selection import train_test_split
import random


import warnings
warnings.filterwarnings("ignore")

#Obtenemos word_embeddings
embeddings_path = '../AmazonWE/sentic2vec.csv'
line_embeddings = CSVEmbeddings(embeddings_path).embeddings

#Definimos tamaño ventana
windows_size = 3

#Fijamos la semilla para poder reproducir experimentos
random.seed(123456)

#Tamaño de test
test_size = 0.3

def create_matrix(sentences, windows_size, word_to_index, labels):
	'''
		Método que crea las matrices con un tamaño de ventana dado
	'''

	#Matrices de ventanas de palabras y etiquetas
	x_matrix = []
	y_vector = []

	padding_index = word_to_index['PADDING']
	unknown_index = word_to_index['UNKOWN']

	n_words = n_unknown = 0
	count_sentence = 0

	for sentence in sentences:
		#índice en la frase
		target_word_idx = 0
		#Para cada una de las palabras en la frase
		for target_word_idx in range(len(sentence)):
			
			wordIndices = []

			#Para cada palabra en la ventana que estamos contemplando -> (?) ¿NO SERÍA EL TAMAÑO DE VENTANA / 2 -1?
			#for index in range(sentence_idx - windows_size//2, sentence_idx + windows_size//2 + 1):
			for word_position in range(target_word_idx - windows_size, target_word_idx + windows_size + 1):
				#Si nos salimos de la frase rellenamos con 0 (padding_index)
				if word_position < 0 or word_position >= len(sentence):
					wordIndices.append(padding_index)
					continue

				word = sentence[word_position][0]
				n_words += 1

				#Buscamos su índice en los embeddings
				if word in word_to_index:
					word_index = word_to_index[word]
				elif word.lower() in word_to_index:
					word_index = word_to_index[word.lower()]
				else:
					word_index = unknown_index
					n_unknown += 1


				wordIndices.append(word_index)


			#Introducimos fila en la matriz
			x_matrix.append(wordIndices)
			y_vector.append(labelsToIdx(labels[count_sentence][target_word_idx]))

		count_sentence = count_sentence + 1

	return (x_matrix, y_vector)



def labelsToIdx(label):

	aux = 0

	if(label == 'O'):
		aux = 1
	elif(label == 'B'):
		aux = 2
	elif(label == 'I'):
		aux = 3

	return aux

##############################################################################################
# 								PREPARACIÓN DE LOS DATOS									 #
##############################################################################################

# Conjunto de datos que vamos a utilizar

data_source = 'Laptop_Train_v2.xml'
data = XMLData(data_source).getIobData()


words = {}
labels = []

#Para cada frase del conjunto de datos
for sentence in data:
	fila_labels = []
	for word, label in sentence:
		fila_labels.append(label)
		words[word] = True

	labels.append(fila_labels)



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
	if word in words:
		vector_aux = np.array([float(num) for num in emb[1:]])
		wordEmbeddings.append(vector_aux)
		word_to_index[word] = len(word_to_index)




wordEmbeddings = np.array(wordEmbeddings)

embeddings = {'wordEmbeddings': wordEmbeddings, 'word2Idx': word_to_index}

##############################################################################################
# 										CREATE NETWORK										 #
##############################################################################################

num_hiddens_units = 300

#Creamos funciones de entrenamiento y predicción
#n_in = windows_size
n_in = windows_size*2 +1
n_out = 4

#Capas
words_input = tf.contrib.keras.layers.Input(shape = (n_in, ), dtype = "int32", name = "words_input")
words = tf.keras.layers.Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1], weights=[wordEmbeddings], trainable=False)(words_input)
words = tf.keras.layers.Flatten()(words)

output = words
output = tf.keras.layers.Dense(units=num_hiddens_units, activation='tanh')(output)
output = tf.keras.layers.Dense(units=n_out, activation='softmax')(output)


#Creamos modelo
model = tf.keras.Model(inputs=words_input, outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
model.summary()


##############################################################################################
# 									PARTICIONAMIENTO										 #
##############################################################################################

#Creamos las particiones de entrenamiento y test a partir de data.

#obtenemos los datos

tokens, labels = create_matrix(data, windows_size, word_to_index, labels)

tokens_train, tokens_test, labels_train, labels_test = train_test_split(tokens, labels, test_size=test_size, shuffle=False)

tokens_train = np.array(tokens_train)
tokens_test = np.array(tokens_test)
labels_train = np.array(labels_train)
labels_test = np.array(labels_test)



##############################################################################################
# 								FUNCIONES CALCULO ERROR										 #
##############################################################################################

'''
def compute_precision(labels_predicted, labels):
    n_correct = 0
    count = 0
    
    idx = 0
    while idx < len(labels_predicted):
        if labels_predicted[idx] == 2: #He encontrado entidad
        	count += 1

        	if labels_predicted[idx] == labels[idx]:
        		idx += 1
        		found = True

        		while idx < len(labels_predicted) and labels_predicted[idx] == 3: #Mientras sigo dentro de la misma entidad
        			if labels_predicted[idx] != labels[idx]:
        				found = False

        			idx += 1

        		if idx < len(labels_predicted):
        			if labels[idx] == 3: #Si la entidad tenía más tokens de los predichos
        				found = False

        		if found: #Sumamos 1 al número de encontrados
        			n_correct += 1

        	else:
        		idx += 1

        else: 
        	idx += 1

    precision = 0
    if count > 0:
    	precision = float(n_correct) / count

    return precision
'''

def compute_precision(labels_predicted, labels):
	n_correct = 0
	count = 0

	for i in range(0, len(labels)):
		if labels_predicted[i] == labels[i]:
			n_correct += 1

		count += 1


	percision = 0
	if count  > 0:
		precision = float(n_correct)/count

	return precision

def compute_f1(predictions, dataset_y): 
    
    prec = compute_precision(predictions, dataset_y)
    rec = compute_precision(dataset_y, predictions)
    
    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);
        
    return prec, rec, f1



##############################################################################################
# 									TRAINING NETWORK										 #
##############################################################################################

number_of_epochs = 10
minibatch_size = 128

def predict_classes(p):
	return p.argmax(axis=-1)

for epoch in range(number_of_epochs):
    print("\n------------- Epoch %d ------------" % (epoch+1))
    model.fit(tokens_train, labels_train, epochs=1, batch_size=minibatch_size, verbose=True, shuffle=True)   
    
    # Compute precision, recall, F1 on dev & test data
    pre_test, rec_test, f1_test = compute_f1(predict_classes(model.predict([tokens_test])), labels_test)

    print("%d. epoch:  F1 on test: %f" % (epoch+1, f1_test))









