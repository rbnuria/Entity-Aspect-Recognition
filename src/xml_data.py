######################################################################################
#
#	Clase que lee un archivo XML y obtiene todas las frases que contiene almacenándolas
#	en un dato miembro de la clase.
#	
#	Autora: Nuria Rodríguez Barroso (https://github.com/rbnuria)
#
######################################################################################

#-*- coding: UTF-8 -*-

import xml.etree.ElementTree as ET
from nltk import word_tokenize, pos_tag, ne_chunk

class XMLData:

	def __init__(self):
		self.data = []
		self.id = []
		self.aspect = []
		self.tokenized_data = []
		self.iob_data = []
	
	def __init__(self, source):
		tree = ET.parse(source)
		self.data = []
		self.id = []
		self.aspect = []
		self.iob_data = []

		for sentence in tree.iter('sentence'):
			text = sentence.find('text').text
			_id = sentence.attrib['id']
			aspectTerms = sentence.find('aspectTerms')

			aspect_ = []

			if aspectTerms is not None:
				for aspectTerm in aspectTerms.findall('aspectTerm'):
					tokens = word_tokenize(aspectTerm.attrib['term'])
					for token in tokens:
						aspect_.append(token)

			self.data.append(text)
			self.id.append(_id)
			self.aspect.append(aspect_)
			self.iob_data.append(self.iobNotation(text, aspect_))


	def iobNotation(self, phrase, aspects):

		begin = False
		phrase = word_tokenize(phrase)
		
		tokenized_aspects = []

		iob_dataset = []

		for word in phrase:
			if word in aspects:
				if(begin == False):
					iob_dataset.append((word, 'B'))
					begin = True
				else:
					iob_dataset.append((word, 'I'))
			else:
				iob_dataset.append((word, 'O'))
				if(begin == True):
					begin = False


		return iob_dataset


	def getIobData(self):
		return self.iob_data



if __name__ == "__main__":
	source = '/Users/nuria/TFG/SemEval14/Laptop_Train_v2.xml'
	object_ = XMLData(source).getIobData()

