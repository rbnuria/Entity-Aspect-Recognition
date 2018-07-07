######################################################################################
#
#	Clase que se encarga de obtener los embeddings de un fichero .csv
#	
#	Autora: Nuria Rodríguez Barroso (https://github.com/rbnuria)
#
######################################################################################

import csv 
import bz2 
from gensim.models import KeyedVectors
import pandas as pd

class CSVEmbeddings:

	def __init__(self):
		self.embeddings = []

	def __init__(self, source):
		self.embeddings = []

		with open(source, 'r') as csvfile:
			self.embeddings = list(csv.reader(csvfile))
			#self.embeddings = [str.split(line) for line in csvfile]

		#table = pd.read_table(source, sep=",", index_col=0, header=None, quoting=csv.QUOTE_NONE)
		 		
		#with bz2.BZ2File(source, 'r') as binfile:
		#	binfile = binfile.read()
		#	self.embeddings = [str.split(line) for line in binfile]

	def getEmbeddings(self, source):
	    '''
	    	Método que se encarga de leer un archivo CSV ubicado en la ruta pasada como argumento.
	    	El método almacenará los embeddings que se encuentran en el CSV introducido.
	    '''

	    return self.embeddings


