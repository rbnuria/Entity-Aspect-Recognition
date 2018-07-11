from csv_embeddings import *
from txt_embeddings import *
from xml_data import *
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


class Model_RRNN:
	def __init__(self,embeddings_path = None, train_path = None, test_path = None, max_length = None, batch_size = None):
		self.embeddings_path = embeddings_path
		self.train_path = train_path
		self.test_path = test_path
		self.max_length = max_length
		self.batch_size = batch_size
		self.x_train = None
		self.x_test = None
		self.y_train = None
		self.y_test = None
		self.predicted_labels = None
		self.word_embeddings = None
		self.sequence_lengths_train = []
		self.sequence_lengths_test = []

		if(embeddings_path != None and train_path != None and test_path != None):
			self.readData()
		

	def readData(self):
		'''
			Método que lee los datos y los prepara.
		'''

		#Lectura de datos
		embeddings = TXTEmbeddings(self.embeddings_path).getEmbeddings()
		print("Leyendo datos de train...")
		self.train = XMLData(self.train_path).getIobData()
		print("Datos de train leídos.")
		print("Leyendo datos de test...")
		self.test = XMLData(self.test_path).getIobData()
		print("Datos de test leídos.")


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

		for sentence in self.train:
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


		for sentence in self.test:
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
		self.sequence_lengths_train = [self.max_length if(len(tokens)>self.max_length) else len(tokens) for tokens in self.train]
		self.sequence_lengths_test = [self.max_length if(len(tokens)>self.max_length) else len(tokens) for tokens in self.test]

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


	#FUNCIONES DE CÁLCULO DEL ERROR

	def compute_custom_recall(self,labels_predicted, labels):
		'''
			Función que calcula la precisión como entidades_encontradas / entidades_reales
		'''

		n_correct = 0
		count = 0
		indice = 0
		idx = 0
		precision_total = 0

		for indice in range(0, labels_predicted.shape[0]):
			label_predicted = labels_predicted[indice]
			label = labels[indice]

			idx = 0

			for idx in range(0, len(label)):
				if label[idx] == 2: #He encontrado entidad
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

	def compute_precision(self, labels_predicted, labels):
		total_precision = 0
		for i in range(0, labels.shape[0]):
			total_precision += sklearn.metrics.precision_score(self.predicted_labels[i], self.y_test[i], average = 'macro')

		return total_precision/labels.shape[0]

	def compute_custom_precision(self,labels_predicted, labels):
		'''
			Función que calcula el recall como entidades_encontradas / entidades etiquetadas
		'''

		n_correct = 0
		count = 0
		indice = 0
		idx = 0
		precision_total = 0

		for indice in range(0, labels.shape[0]):
			label_predicted = labels_predicted[indice]
			label = labels[indice]

			idx = 0

			for idx in range(0, len(label_predicted)):
				if label_predicted[idx] == 2: #He etiquetado entidad
					count += 1

					if label[idx] == label_predicted[idx]:
						idx += 1
						found = True

						while idx < len(label) and label_predicted[idx] == 3: #Mientras sigo dentro de la misma entidad
							if label[idx] != label_predicted[idx]:
								found = False

							idx += 1

						if idx < len(label):
							if label_predicted[idx] == 3: #Si la entidad tenía más tokens de los predichos
								found = False

						if found: #Sumamos 1 al número de encontrados
							n_correct += 1

					else:
						idx += 1
				else:
					idx += 1

		return (float(n_correct)/count if count > 0 else 0)


	def compute_recall(self, labels_predicted, labels):
		total_precision = 0
		for i in range(0, labels.shape[0]):
			total_precision += sklearn.metrics.recall_score(self.predicted_labels[i], self.y_test[i], average = 'macro')

		return total_precision/labels.shape[0]

	def compute_custom_f1(self, labels_predicted, labels):
		p = self.compute_custom_precision(labels_predicted, labels)
		r = self.compute_custom_recall(labels_predicted, labels)

		return (2*(p*r)/(p+r) if (p+r) != 0 else 0)

	def compute_f1(self, labels_predicted, labels):
		p = self.compute_precision(labels_predicted, labels)
		r = self.compute_recall(labels_predicted, labels)


		return (2*(p*r)/(p+r) if (p+r) != 0 else 0)


	def calculateAccuracy(self):
		print("*************************** RESULTADOS **************************")
		print("PRECISION\tRECALL\tF1")
		print("ETIQUETADO PARCIAL:" )
		print(str(self.compute_precision(self.predicted_labels, self.y_test)) + "\t" + str(self.compute_recall(self.predicted_labels, self.y_test)) + "\t" + str(self.compute_f1(self.predicted_labels, self.y_test)))
		print
		print("ETIQUETADO TOTAL:" )
		print(str(self.compute_custom_precision(self.predicted_labels, self.y_test)) + "\t" + str(self.compute_custom_recall(self.predicted_labels, self.y_test)) + "\t" + str(self.compute_custom_f1(self.predicted_labels, self.y_test)))


	def getAspects(self, frase):
		indice = 0
		aspectos = []


		while indice < len(self.predicted_labels[frase]):
			label = self.predicted_labels[frase][indice]
			
			#Si empieza una entidad
			if(label == 2):
				aspecto = self.test[frase][indice][0]

				indice += 1
				label = self.predicted_labels[frase][indice]

				while(label == 3):
					aspecto += " " + self.test[frase][indice][0]
					indice += 1
					label = self.predicted_labels[frase][indice]

				aspectos.append(aspecto)


			indice += 1

		return aspectos
	
	def saveData(self, source):
		with open(source, "w") as xmlfile:
			xmlfile.write('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n')
			xmlfile.write('<sentences>\n')
			count = 0
			#Recorremos las oraciones del test
			for sentence in self.test:
				xmlfile.write('\t<sentence>\n')

				#Texto
				frase = ""
				for word, label in sentence:
					frase += word + " "

				xmlfile.write('\t\t<text>' + frase + '</text>\n')

				#Aspectos
				aspectos = self.getAspects(count)
				if aspectos != []:
					xmlfile.write('\t\t<aspectTerms>\n')
					for asp in aspectos:
						xmlfile.write('\t\t\t<aspectTerm term="' + asp + '" />\n')

					xmlfile.write('\t\t</aspectTerms>\n')


				xmlfile.write('\t</sentence>\n')
				count += 1


			xmlfile.write('</sentences>')

	def to_json(self):
		return self.model.to_json()

	def save_weights(self,source):
		self.model.save_weights()

	def save(self, source):
		self.model.save(source)


	def save_weights(self, source):
		self.model.save_weights(source)
