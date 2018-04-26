######################################################################################
#
#	Clase que lee un archivo .txt con una estructura concreta y obtiene 
#	todas las frases que contiene almacenándolas en un dato miembro de la clase.
#	
#	Autora: Nuria Rodríguez Barroso (https://github.com/rbnuria)
#
######################################################################################

#-*- coding: UTF-8 -*-

import numpy as np 

class TXTData:

	def __init__(self, source):
		'''	
			Constructor que inicializa por defecto los aributos de la clase.

			En cuanto al texto del archivo:
			Se encarga de leer un archivo .txt ubicado en la ruta pasada como argumento.
			El método almacena en self.text el contenido de este archivo para ser posteriormente
			manipulado amlacenando la información que nos convenga. 

		'''
		self.data = []

		f = open(source, 'r')
		self.text = f.read()
		f.close()

	def __init__(self):
		'''	
			Constructor que inicializa por defecto los aributos de la clase.
		'''
		self.data = []
		self.text = []

	def getData(self, source):
	    '''
	    	Método que se encarga de leer un archivo .txt ubicado en la ruta pasada como argumento.
	    	El método almacenará la lista de frases de las que se compone el archivo xml introducido.
	    '''
	    
	    new = True

	    for line in open(source, 'r'):
	    	if '[t]' in line and new == True:
	    		#Hacemos comprobación para no meter \n en el titulo
	    		#También se le puede quitar el [t] pero no me parece necesario
	    		if line[len(line)-1] == '\n':
	    			title = line[:len(line)-1]
	    		else:
	    			title = line
	    		new = False
	    		text = ""
	    	elif '[t]' in line and new == False:
	    		self.data.append((title, text))
	    		if line[len(line)-1] == '\n':
	    			title = line[:len(line)-1]
	    		else:
	    			title = line
	    		text = ""
	    	elif new == False and '##' in line:
	    		pos = line.find('##')
	    		if line[len(line)-1] == '\n':
	    			text = text + line[pos+2:len(line)-1]
	    		else:
	    			text = text + line[pos+2:len(line)]


	    return self.data
	    	

'''if __name__ == "__main__":
	source = '/Users/nuria/TFG/customer review data/Nokia 6610.txt'
	object_ = TXTData()
	data = object_.getData(source)
	print(data[10])
'''
