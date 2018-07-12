from txt_embeddings import *
from trainer import *
from predictor import *

if __name__ == "__main__":
	#Leemos embeddings

	txt_embeddings = TXTEmbeddings("../glove.840B.300d.word2vec.txt")

	#Guardamos embeddings para no terner que volver a cargarlos
	embeddings = txt_embeddings.getEmbeddings()
	vocabulary = txt_embeddings.getVocabulary()


	#Entrenamos modelo
	trainer = Trainer("conf.csv", embeddings, vocabulary)

	#Guardamos parámetros de red
	trainer.save()

	#Guardamos parámetros de transición
	transition_params = trainer.getTransitionParams()


	#Predecimos el modelo tras cargarlo
	predictor = Predictor("conf.csv", embeddings, vocabulary, transition_params)
	predictor.predict()