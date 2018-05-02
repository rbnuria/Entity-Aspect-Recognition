import numpy as np
from csv_embeddings import *
from xml_data import *

#Obtenemos word_embeddings
embeddings_path = '../AmazonWE/sentic2vec.csv'
line_embeddings = CSVEmbeddings(embeddings_path).embeddings

#Definimos tamaño ventana
windows_size = 3

#Creo que los word_embeddings están entre 0-1 -> Esto habría que hacerlo de forma automática mejor




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
			for index in range(sentence_idx - windows_size, sentence_idx + windows_size + 1):

				#Si nos salimos de la frase rellenamos con 0 (padding_index)
				if index < 0 or index >= len(sentence):
					wordIndices.append(padding_index)

				word = sentence[index][0]
				n_word += 1

				#Buscamos su índice en los embeddings
				if word in embeddings:
					word_index = embeddings[word]
				else:
					word_index = unknown_index
					n_unknown += 1


			#Hacemos ahora el vector de etiquetas con los índices 
			label_index = label_to_index[sentence[word_index][1]]

			#Introducimos fila en la matriz
			x_matrix.append(wordIndices)
			y_label.append(label_index)




##############################################################################################
# 								PREPARACIÓN DE LOS DATOS									 #
##############################################################################################

# Conjunto de datos que vamos a utilizar

data_source = 'Laptop_Train_v2.xml'
data = XMLData(data_source).getIobData()

# PARTICIONAR EL CONJUNTO -> HACER DATA = CONJUNTO TERAIN + CONJUNTO TEST

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

embeddings = {'wordEmbeddings': wordEmbeddings, 'word2Idx': word2Idx,'label2Idx': label2Idx}








