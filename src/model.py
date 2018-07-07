from csv_embeddings import *
from txt_embeddings import *
from xml_data import *
import numpy as np
from sklearn.model_selection import train_test_split


class Model_RRNN:
	def __init__(self,embeddings_path, data_path, max_length, batch_size, test_size):
		self.embeddings_path = embeddings_path
		self.data_path = data_path
		self.max_length = max_length
		self.batch_size = batch_size
		self.test_size = test_size
		self.x_train = None
		self.x_test = None
		self.y_train = None
		self.y_test = None
		self.word_embeddings = None
		self.sequence_lengths_train = []
		self.sequence_lengths_test = []

		self.readData()
		

	def readData(self):
		line_embeddings = CSVEmbeddings(self.embeddings_path).embeddings
		data = XMLData(self.data_path).getIobData()

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
				vector_aux = np.zeros(300)
				wordEmbeddings.append(vector_aux)


				word_to_index["UNKOWN"] = len(word_to_index)
				vector_aux = np.random.uniform(-0.25, 0.25, 300)
				wordEmbeddings.append(vector_aux)


			#Si la palabra está en nuestro diccionario (caso concreto que estemos ejecutando) -> Lo añadimos a wordEmbeddings
			if word.lower() in words:
				vector_aux = np.array([float(num) for num in emb[1:]])
				wordEmbeddings.append(vector_aux)
				word_to_index[word] = len(word_to_index)


		wordEmbeddings = np.array(wordEmbeddings)
		embeddings = {'wordEmbeddings': wordEmbeddings, 'word2Idx': word_to_index}


		#tokens_train, tokens_test, labels_train, labels_test = train_test_split(sentences, labels, test_size=self.test_size, shuffle=False)
		#tokens_train, labels_train = self.fixTrainSize(tokens_train, labels_train)

		#self.sequence_lengths_train = [self.max_length if(len(tokens)>self.max_length) else len(tokens) for tokens in tokens_train]
		#self.sequence_lengths_test = [self.max_length if(len(tokens)>self.max_length) else len(tokens) for tokens in tokens_test]


		#tokens_train, labels_train = self.dataPreparation(tokens_train, labels_train, word_to_index)
		#tokens_test, labels_test = self.dataPreparation(tokens_test, labels_test, word_to_index)

		#tokens_train = np.array(tokens_train)
		#tokens_test = np.array(tokens_test)
		#labels_train = np.array(labels_train)
		#labels_test = np.array(labels_test)

		tokens_train, tokens_test, labels_train, labels_test = train_test_split(sentences, labels, test_size=self.test_size, shuffle=False)
		tokens_train, labels_train = self.fixTrainSize(tokens_train, labels_train)
		
		tokens_train_prepared, labels_train_prepared = self.dataPreparation(tokens_train, labels_train, word_to_index)
		tokens_test_prepared, labels_test_prepared = self.dataPreparation(tokens_test, labels_test, word_to_index)
		
		self.sequence_lengths_train = [self.max_length if(len(tokens)>self.max_length) else len(tokens) for tokens in tokens_train]
		self.sequence_lengths_test = [self.max_length if(len(tokens)>self.max_length) else len(tokens) for tokens in tokens_test]

		tokens_train_prepared = np.array(tokens_train_prepared)
		tokens_test_prepared = np.array(tokens_test_prepared)
		labels_train_prepared = np.array(labels_train_prepared)
		labels_test_prepared = np.array(labels_test_prepared)


		data = {
		    'wordEmbeddings': wordEmbeddings, 'word2Idx': word_to_index,
		    'train': {'sentences': tokens_train_prepared, 'labels': labels_train_prepared},
		    'test':  {'sentences': tokens_test_prepared, 'labels': labels_test_prepared}
		    }

		self.word_embeddings = data['wordEmbeddings']

		self.x_train = data['train']['sentences']
		self.x_test = data['test']['sentences']

		self.y_train = data['train']['labels']
		self.y_test = data['test']['labels']

		print(self.x_train[0])
		print(self.x_train[1])
		print(self.x_train[2])
		print(self.x_train[3])
		print(self.x_train[4])


		print("************************")

		print(self.y_train[0])
		print(self.y_train[1])
		print(self.y_train[2])
		print(self.y_train[3])
		print(self.y_train[4])



		print('X_train shape:', self.x_train.shape)
		print('X_test shape:', self.x_test.shape)
		print('Y_train shape:', self.y_train.shape)
		print('Y_test shape;', self.y_test.shape)


	def longerSentence(data):
		max_sentence_length = 0
		for sentence in data:
			max_sentence_length = max(max_sentence_length, len(sentence))

		return max_sentence_length


	def padding_truncate(self,training_sentences):
	    """Amplia o recorta las oraciones de entrenamiento.
	    
	    Args:
	        training_sentences: Lista de listas de enteros.

	    -> En este caso solo ampliaría pues estamos tomando como tamaño la frase más larga     
	    """
	    
	    for i in range(len(training_sentences)):
	        sent_size = len(training_sentences[i])
	        if sent_size > self.max_length:
	            training_sentences[i] = training_sentences[i][:self.max_length]
	        elif sent_size < self.max_length:
	            training_sentences[i] += [0] * (self.max_length - sent_size)
	    	
	    return training_sentences 
	            


	def labelsToIdx(self,label):

		aux = 0

		if(label == 'O'):
			aux = 1
		elif(label == 'B'):
			aux = 2
		elif(label == 'I'):
			aux = 3

		return aux

	def fixTrainSize(self,train, labels):
		'''
			Método para aumentar el train repitiendo frases hasta llegar al primer número múltiplo de batch_size
		'''

		resto = len(train) % self.batch_size
		#Número de frases que tenemos que añadir
		num_sentences = self.batch_size-resto

		#Elegimos estas frases de forma aleatoria (con distribución uniforme)
		for i in range(0, num_sentences):
			r = int(np.random.uniform(0, len(train)-1,1))
			train.append(train[r])
			labels.append(labels[r])

		return train, labels

	def dataPreparation(self,sentences, labels, word_to_index):
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
		x_matrix = self.padding_truncate(x_matrix)


		for fila_labels in labels:
			labelIndices = []

			for label in fila_labels:
				labelIndices.append(self.labelsToIdx(label))

			y_matrix.append(labelIndices)

		#Padding a las etiquetas
		y_matrix = self.padding_truncate(y_matrix)

		return (x_matrix, y_matrix)

	def getLabelsTest(self):
		return self.y_test;



