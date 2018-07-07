######################################################################################
#
#	Clase que se encarga de obtener los embeddings de un fichero .txt con pandas
#	
#	Autora: Nuria Rodríguez Barroso (https://github.com/rbnuria)
#
######################################################################################

import csv 
import pandas as pd


class TXTEmbeddings:

	def __init__(self):
		self.embeddings = []

	def __init__(self, source):

		with open(source, 'r') as csvfile:
			self.embeddings = [str.split(line) for line in csvfile]


	def getEmbeddings(self, source):
	    '''
	    	Devuelve el dato miembro self.embeddings
	    '''

	    return self.embeddings



