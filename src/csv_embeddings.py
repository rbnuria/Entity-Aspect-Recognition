######################################################################################
#
#	Clase que se encarga de obtener los embeddings de un fichero .csv
#	
#	Autora: Nuria Rodríguez Barroso (https://github.com/rbnuria)
#
######################################################################################

import csv 

class CSVEmbeddings:

	def __init__(self):
		self.embeddings = []

	def getEmbeddings(self, source):
	    '''
	    	Método que se encarga de leer un archivo CSV ubicado en la ruta pasada como argumento.
	    	El método almacenará los embeddings que se encuentran en el CSV introducido.
	    '''
	    with open(source, 'r') as csvfile:
	    	self.embeddings = list(csv.reader(csvfile))

	    return self.embeddings

if __name__ == "__main__":
	object_ = CSVEmbeddings()
	source = '/Users/nuria/TFG/AmazonWE/sentic2vec.csv'
	embeddings = object_.getEmbeddings(source)


