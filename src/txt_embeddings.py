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

class TXTEmbeddings:

	def __init__(self):
		self.embeddings = []

	def __init__(self, source):

		print("Leyendo embeddings...")

		#with open(source, 'r') as txtfile:
		#	self.embeddings = [str.split(line) for line in txtfile]
		#	print("Embeddings leídos.")

		self.embeddings = KeyedVectors.load_word2vec_format(source, binary = False, limit = 200000)

		print("Embeddings leídos.")


	def getEmbeddings(self):
	    '''
	    	Devuelve el dato miembro self.embeddings
	    '''

	    return self.embeddings



