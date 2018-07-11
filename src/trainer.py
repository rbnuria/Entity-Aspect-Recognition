import model
from cnn_model import *
from lstm_model import *
import csv
from keras.models import model_from_json

class Trainer:
	def __init__(self, conf_file):
		'''
			Constructor de la clase Trainer que entrena un modelo y lo
			guarda en un función de los datos del fichero de configuración
		'''

		with open(conf_file) as csvfile:
			reader = csv.DictReader(csvfile)

			self.conf_data = {}

			for row in reader:
				self.conf_data[row['dato']] = row['valor']
				


	def configure(self):

		embeddings_path = self.conf_data['path_embeddings']
		train_path = self.conf_data['path_train']
		test_path = self.conf_data['path_test']
		max_length = int(self.conf_data['max_length'])
		batch_size = int(self.conf_data['batch_size'])
		
		if(self.conf_data['type'] == "cnn"):
			filtro_1 = int(self.conf_data['filtro_1'])
			filtro_2 = int(self.conf_data['filtro_2'])
			kernel_1 = int(self.conf_data['kernel_1'])
			kernel_2 = int(self.conf_data['kernel_2'])
			dropout_1 = float(self.conf_data['dropout_1'])
			dropout_2 = float(self.conf_data['dropout_2'])
			
			#Creamos modelo
			self.model = CNN_Model(embeddings_path, train_path, test_path, max_length, batch_size, filtro_1, filtro_2, kernel_1, kernel_2, dropout_1, dropout_2)
		else:
			num_units = float(self.conf_data['units_lstm'])

			self.model = LSTM_Model(embeddings_path, train_path, test_path, max_length, batch_size, num_units)

		

	def save(self):
		self.configure()
		print("Creación del modelo...")
		self.model.trainModel()
		self.model.fitModel(int(self.conf_data['epoch']))
		print("Modelo creado.")
		#Guardamos el modelo

		print("Guardando el modelo...")
		name_json = self.conf_data['name_json']

		model_json = self.model.to_json()
		with open(name_json, "w") as json_file:
			json_file.write(model_json)


		self.model.save_weights(self.conf_data['name_h5'])
		print("Modelo guardado.")


if __name__ == "__main__":
	trainer = Trainer("conf.csv")

	trainer.save()



