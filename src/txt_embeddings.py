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

		self.embeddings = KeyedVectors.load_word2vec_format(source, binary = False)

		print("Embeddings leídos")


	def getEmbeddings(self, source):
	    '''
	    	Devuelve el dato miembro self.embeddings
	    '''

	    return self.embeddings



