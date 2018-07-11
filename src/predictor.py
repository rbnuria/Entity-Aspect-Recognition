import model
from cnn_model import *
from lstm_model import *
from combined_model import *
import csv

class Predictor:
	def __init__(self, conf_file):
		'''
			Constructor de la clase Trainer que entrena un modelo y lo
			guarda en un función de los datos del fichero de configuración
		'''

		with open(conf_file) as csvfile:
			reader = csv.DictReader(csvfile)

			conf_data = {}

			for row in reader:
				conf_data[row['dato']] = row['valor']


			source = conf_data['save_model_path']
			type_ = conf_data['type']
			




if __name__ == "__main__":
	trainer = Predictor("conf.csv")



