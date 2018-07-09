from csv_embeddings import *
from txt_embeddings import *
from xml_data import *
import numpy as np
from sklearn.model_selection import train_test_split


class Model_RRNN:
	def __init__(self,embeddings_path, train_path, test_path, max_length, batch_size, test_size):
		self.embeddings_path = embeddings_path
		self.train_path = train_path
		self.test_path = test_path
		self.max_length = max_length
		self.batch_size = batch_size
		self.test_size = test_size
		self.x_train = None
		self.x_test = None
		self.y_train = None
		self.y_test = None
		self.predicted_labels = None
		self.word_embeddings = None
		self.sequence_lengths_train = []
		self.sequence_lengths_test = []

		self.readData()
		

	def readData(self):
		'''
			Método que lee los datos y los prepara.
		'''

		#Lectura de datos
		embeddings = TXTEmbeddings(self.embeddings_path).getEmbeddings()
		train = XMLData(self.train_path).getIobData()
		test = XMLData(self.test_path).getIobData()


		#Preparación vocabulario y embeddings_matrix
		vocabulary = {}
		vocabulary["PADDING"] = len(vocabulary)
		vocabulary["UNKOWN"] = len(vocabulary)

		embeddings_matrix = []
		embeddings_matrix.append(np.zeros(300))
		embeddings_matrix.append(np.random.uniform(-0.25, 0.25, 300))


		for word in embeddings.wv.vocab:
			vocabulary[word] = len(vocabulary)
			#Al mismo tiempo creamos matrix de embeddings
			embeddings_matrix.append(embeddings[word])

		#Word2Idx a test y train
		train_idx = []
		ltrain_idx = []
		test_idx = []
		ltest_idx = []

		for sentence in train:
			wordIndices = []
			labelIndices = []
			for word, label in sentence:
				#Si la palabra está en el vocabulario, asignamos su índice en él
				if word in vocabulary:
					wordIndices.append(vocabulary[word])
				else:
					#Padding
					if word == "-":
						wordIndices.append(vocabulary["PADDING"])
					#Desconocida
					else:
						wordIndices.append(vocabulary["UNKOWN"])

				labelIndices.append(self.labelsToIdx(label))


			ltrain_idx.append(np.array(labelIndices))
			train_idx.append(np.array(wordIndices))


		for sentence in test:
			wordIndices = []
			labelIndices = []
			for word, label in sentence:
				#Si tenemos embedding para la palabra
				if word in vocabulary:
					wordIndices.append(vocabulary[word])
				else:
					#Padding
					if word == "-":
						wordIndices.append(vocabulary["PADDING"])
					#Desconocida
					else:
						wordIndices.append(vocabulary["UNKOWN"])

				labelIndices.append(self.labelsToIdx(label))

			ltest_idx.append(np.array(labelIndices))
			test_idx.append(np.array(wordIndices))


		#Guardamos longitudes 
		self.sequence_lengths_train = [self.max_length if(len(tokens)>self.max_length) else len(tokens) for tokens in train]
		self.sequence_lengths_test = [self.max_length if(len(tokens)>self.max_length) else len(tokens) for tokens in test]

		#Padding a datos y etiquetas
		tokens_train_prepared = np.array(self.padding_truncate(train_idx))
		tokens_test_prepared = np.array(self.padding_truncate(test_idx))
		labels_train_prepared = np.array(self.padding_truncate(ltrain_idx))
		labels_test_prepared = np.array(self.padding_truncate(ltest_idx))


		data = {
		    'wordEmbeddings': np.array(embeddings_matrix), 'word2Idx': vocabulary,
		    'train': {'sentences': tokens_train_prepared, 'labels': labels_train_prepared},
		    'test':  {'sentences': tokens_test_prepared, 'labels': labels_test_prepared}
		    }



		self.word_embeddings = data['wordEmbeddings']

		self.x_train = data['train']['sentences']
		self.x_test = data['test']['sentences']

		self.y_train = data['train']['labels']
		self.y_test = data['test']['labels']

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
	        	if(isinstance(training_sentences[i], list)):
		        	training_sentences[i] += [0] * (self.max_length - sent_size)
		        else:
		        	list_sentence = training_sentences[i].tolist()
		        	list_sentence += [0] * (self.max_length - sent_size)
		        	training_sentences[i] = np.array(list_sentence)

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
		#y_matrix = self.padding_truncate(y_matrix)

		return (x_matrix, y_matrix)

	def getLabelsTest(self):
		return self.y_test

	def getLabelsTrain(self):
		return self.y_train

	def compute_precision(self,labels_predicted, labels):
		precision = 0
		for j in range(0, labels.shape[0]):
			label = labels[j]
			label_predicted = labels_predicted[j]
			n_correct = 0
			count = 0
			for i in range(0, len(label)):
				if label_predicted[i] == label[i]:
					n_correct += 1

				count += 1

			if count  > 0:
				precision+= float(n_correct)/count

		return precision/labels.shape[0]

	def compute_custom_precision(self,labels_predicted, labels):
	    n_correct = 0
	    count = 0
	    indice = 0
	    idx = 0
	    precision_total = 0

	    for indice in range(0, labels_predicted.shape[0]):
	    	label_predicted = labels_predicted[indice]
	    	label = labels[indice]

	    	idx = 0
	    	
	    	for idx in range(0, len(label_predicted)):
		        if label_predicted[idx] == 2: #He encontrado entidad
		        	count += 1

		        	if label_predicted[idx] == label[idx]:
		        		idx += 1
		        		found = True

		        		while idx < len(label_predicted) and label_predicted[idx] == 3: #Mientras sigo dentro de la misma entidad
		        			if label_predicted[idx] != label[idx]:
		        				found = False

		        			idx += 1

		        		if idx < len(label_predicted):
		        			if label[idx] == 3: #Si la entidad tenía más tokens de los predichos
		        				found = False

		        		if found: #Sumamos 1 al número de encontrados
		        			n_correct += 1

		        	else:
		        		idx += 1

		        else: 
		        	idx += 1
	    
	    return (float(n_correct)/count if count > 0 else 0)


	def calculateAccuracy(self):
		print("**************************************")
		print(str(self.compute_precision(self.predicted_labels, self.y_test)) + "\t" + str(self.compute_custom_precision(self.predicted_labels, self.y_test)))










