######################################################################################
#
#	Clase que se encarga de obtener los embeddings de un fichero .txt con pandas
#	
#	Autora: Nuria Rodríguez Barroso (https://github.com/rbnuria)
#
######################################################################################

import csv 
import pandas as pd
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import word2vec
import numpy as np

class TXTEmbeddings:

	def __init__(self):
		self.embeddings = []

	def __init__(self, source):

		print("Leyendo embeddings...")

		#with open(source, 'r') as txtfile:
		#	self.embeddings = [str.split(line) for line in txtfile]
		#	print("Embeddings leídos.")

		embeddings = KeyedVectors.load_word2vec_format(source, binary = False, limit = 200000)


		#Preparación vocabulario y embeddings_matrix
		self.vocabulary = {}
		self.vocabulary["PADDING"] = len(self.vocabulary)
		self.vocabulary["UNKOWN"] = len(self.vocabulary)

		self.embeddings_matrix = []
		self.embeddings_matrix.append(np.zeros(300))
		self.embeddings_matrix.append(np.random.uniform(-0.25, 0.25, 300))


		for word in embeddings.wv.vocab:
			self.vocabulary[word] = len(self.vocabulary)
			#Al mismo tiempo creamos matrix de embeddings
			self.embeddings_matrix.append(embeddings[word])

		print("Embeddings leídos.")


	def getEmbeddings(self):
	    '''
	    	Devuelve la matriz de embeddings
	    '''

	    return np.array(self.embeddings_matrix)


	def getVocabulary(self):
		'''
			Devuelve el vocabulario
		'''

		return self.vocabulary


