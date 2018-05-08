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

#Fijamos la semilla
random.seed(123456)

#Tamaño de test
test_size = 0.3


def create_matrix(sentences, windows_size, word_to_index, label_to_index):
	''' 
	Método que crea las matrices con un tamaño de ventana dado
	'''
	x_matrix = []
	y_vector = []

	padding_index = word_to_index['PADDING']
	unknown_index = word_to_index['UNKOWN']

	n_words = n_unknown = 0

	for sentence in sentences:
		#índice en la frase
		word_index = 0

		#Para cada una de las palabras en la frase
		for sentence_idx in range(len(sentence)):
			
			wordIndices = []

			#Para cada palabra en la ventana que estamos contemplando -> (?) ¿NO SERÍA EL TAMAÑO DE VENTANA / 2 -1?
			for index in range(sentence_idx - windows_size//2, sentence_idx + windows_size//2 + 1):
				#Si nos salimos de la frase rellenamos con 0 (padding_index)
				if index < 0 or index >= len(sentence):
					wordIndices.append(padding_index)
					continue

				word = sentence[index][0]
				n_words += 1

				#Buscamos su índice en los embeddings
				if word in word_to_index:
					word_index = word_to_index[word]
				else:
					word_index = unknown_index
					n_unknown += 1

				wordIndices.append(word_index)


			#Hacemos ahora el vector de etiquetas con los índices 
			label_index = label_to_index[sentence[sentence_idx][1]]

			#Introducimos fila en la matriz
			x_matrix.append(wordIndices)
			y_vector.append(label_index)


	return (x_matrix, y_vector)





##############################################################################################
# 								PREPARACIÓN DE LOS DATOS									 #
##############################################################################################

# Conjunto de datos que vamos a utilizar

data_source = 'Laptop_Train_v2.xml'
data = XMLData(data_source).getIobData()

labelSet = set()
words = {}

#Para cada frase del conjunto de datos
for sentence in data:	
	for word, label in sentence:
		labelSet.add(label)
		words[word] = True


#Crear matriz de mapping para las etiquetas -> asignar enteros (0,1,2) a cada una de las posibles etiqueas (O, B, I)
label_to_index = {}

for label in labelSet:
	label_to_index[label] = len(label_to_index)


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

embeddings = {'wordEmbeddings': wordEmbeddings, 'word2Idx': word_to_index,'label2Idx': label_to_index}

##############################################################################################
# 										CREATE NETWORK										 #
##############################################################################################

num_hiddens_units = 100

#Creamos funciones de entrenamiento y predicción
n_in = windows_size
n_out = len(label_to_index)

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
#model.summary()


##############################################################################################
# 									PARTICIONAMIENTO										 #
##############################################################################################

#Creamos las particiones de entrenamiento y test a partir de data.

#obtenemos los datos

tokens, labels = create_matrix(data, windows_size, word_to_index, label_to_index)

tokens_train, tokens_test, labels_train, labels_test = train_test_split(tokens, labels, test_size=test_size, shuffle=False)

tokens_train = np.array(tokens_train)
tokens_test = np.array(tokens_test)
labels_train = np.array(labels_train)
labels_test = np.array(labels_test)


##############################################################################################
# 								FUNCIONES CALCULO ERROR										 #
##############################################################################################

def compute_precision(labels_predicted, labels):
    n_correct = 0
    count = 0
    
    idx = 0
    while idx < len(labels_predicted):
        if labels_predicted[idx][0] == 'B': #He encontrado entidad
        	count += 1

        	if labels_predicted[idx] == labels[idx]:
        		idx += 1
        		found = True

        		while idx < len(labels_predicted) and labels_predicted[idx][0] == 'I': #Mientras sigo dentro de la misma entidad
        			if labels_predicted[idx] != labels[idx]:
        				found = False

        			idx += 1

        		if idx < len(labels_predicted):
        			if labels[idx][0] == 'I': #Si la entidad tenía más tokens de los predichos
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

def compute_f1(predictions, dataset_y, idx2Label): 
    label_y = [idx2Label[element] for element in dataset_y]
    pred_labels = [idx2Label[element] for element in predictions]
   
    prec = compute_precision(pred_labels, label_y)
    rec = compute_precision(label_y, pred_labels)
    
    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);
        
    return prec, rec, f1



##############################################################################################
# 									TRAINING NETWORK										 #
##############################################################################################

number_of_epochs = 10
minibatch_size = 128

index_to_label = {v: k for k, v in label_to_index.items()}

def predict_classes(p):
	return p.argmax(axis=-1)

for epoch in range(number_of_epochs):
    print("\n------------- Epoch %d ------------" % (epoch+1))
    model.fit(tokens_train, labels_train, epochs=1, batch_size=minibatch_size, verbose=True, shuffle=True)   
    
    
    # Compute precision, recall, F1 on dev & test data
    pre_test, rec_test, f1_test = compute_f1(predict_classes(model.predict([tokens_test])), labels_test, index_to_label)

    print("%d. epoch:  F1 on test: %f" % (epoch+1, f1_test))









