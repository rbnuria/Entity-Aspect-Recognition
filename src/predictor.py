import model
from cnn_model import *
from lstm_model import *
from combined_model import *
import csv

class Predictor:
	def __init__(self, conf_file, embeddings, vocabulary, transition_params):
		'''
			Constructor de la clase Trainer que entrena un modelo y lo
			guarda en un función de los datos del fichero de configuración
		'''
		self.embeddings = embeddings
		self.vocabulary = vocabulary
		self.transition_params = transition_params

		with open(conf_file) as csvfile:
			reader = csv.DictReader(csvfile)

			conf_data = {}

			for row in reader:
				conf_data[row['dato']] = row['valor']

			source = conf_data['save_model_path']
			self.results_source = conf_data['save_results_path']
			type_ = conf_data['type']
			train_path = conf_data['path_train']
			test_path = conf_data['path_test']
			max_length = int(conf_data['max_length'])
			batch_size = int(conf_data['batch_size'])

		if type_ == "cnn":
			filter_1 = int(conf_data['filtro_1'])
			filter_2 = int(conf_data['filtro_2'])
			kernel_1 = int(conf_data['kernel_1'])
			kernel_2 = int(conf_data['kernel_2'])
			dropout_1 = float(conf_data['dropout_1'])
			dropout_2 = float(conf_data['dropout_2'])

			self.model = CNN_Model(self.embeddings, self.vocabulary, train_path, test_path, max_length, batch_size, filter_1, filter_2, kernel_1, kernel_2, dropout_1, dropout_2)
			
			self.model.load_weights(source)

	def predict(self):
		pred = self.model.predictModel(self.transition_params)
		self.model.calculateAccuracy()

		self.model.saveData(self.results_source)
		

