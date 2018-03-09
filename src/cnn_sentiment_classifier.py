'''
Created on 23 feb. 2018

Ejemplo de clasificador opniones a nivel de oración (documento) usando una CNN,
y utilizando la librería TensorFlow.

Se utilizará el corpus sentence_polarity, que en concreto es el corpus de Bing Liu.
NLTK lo ofrece segmentado en oraciones y tokenizado. 
 
@author: Eugenio Martínez Cámara
@organization: Universida de Granada
@requires: TensorFlow, Pythnon3, NLTK, sentence_polarity corpus de NLTK
'''

import random
import statistics
from nltk.corpus import sentence_polarity as sent_pol
from sklearn.model_selection import train_test_split
import tensorflow as tf
from csv_embeddings import *

#Variables globales
RANDOM_SEED = 7
MAX_LENGTH = 65
EMBEDDINGS_DIMENSIONS = 300
KERNEL_SIZE = 2
CNN_OUTPUT_FEATURES = 100
CNN2_OUTPUT_FEATURES = 50
EPOCHS = 10
SIZE_BATCHES = 64 #?
OOV_INDEX = 0

#No está semilla fijada
RANDOM_EMBEDDING = np.random.uniform(low = 1.0, high = 2.0, size = 300)



def loadAmazonWE(filename):
    '''
        Lectura de los embeddings y generación del vocabulario de un archivo pasado como argumento. 

        Returns:
            vocab: vocabulario generado
            embd: word embeddings
    '''
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    file.close()
    return vocab,embd

def data_preparation_train_test():
    """Generación del conjunto de entrenamiento y test
    
    Returns:
        train_sents: es una lista con las oraciones (listas de palabras) para el entrenamiento.
        test_sents: es una lista con las oraciones (listas de palabras) para el test.
        train_labels: lista de enteros con la clase (0:negativo, 1:positivo) de 
        las oraciones de entrenamiento.
        test_labels: lista de enteros con la clase (0:negativo, 1:positivo) de 
        las oraciones de test.
    """
    #Esto lo hago porque a través de la documentación de NLTK sé como es el corpus.
    positive_sents = sent_pol.sents(categories="pos")
    n_pos_sents = len(positive_sents)
    negative_sents = sent_pol.sents(categories="neg")
    n_neg_sents = len(negative_sents)
    #db_indexes: Cada posición se corresponde con una oración del corpus
    db_indexes = [i for i in range(n_pos_sents + n_neg_sents)]
    db_sents = positive_sents + negative_sents
    #db_labels: Cada posición se corresponde con una etiqueta de opinión del corpus.
    #Cada posición de esta lista se corresponde con cada posición de db_indexes.
    db_labels = [1] * n_pos_sents + [0] * n_neg_sents
    train_indexes, test_indexes, train_labels, test_labels = train_test_split(db_indexes, db_labels,test_size=0.2,shuffle=True, stratify=db_labels)
    
    train_sents = [db_sents[i] for i in train_indexes]
    test_sents = [db_sents[i] for i in test_indexes]
    
    return (train_sents, test_sents, train_labels, test_labels)
    
def build_vocabulary(input_corpus, index_start):
    """Genera un vocabulario a partir de un conjunto de oraciones/documentos de
    entrada.
    
    En este caso, las oraciones/documentos deben estar tokenizdos.
    
    Args:
        input_corpus: Lista de listas de oraciones tokenizadas.
        index_start: interger with the first value of the index of the vocabulary.
    """
    vocabulary = {}
    own_lower = str.lower
    index = index_start
    for sent in input_corpus:
        for word in sent:
            word = own_lower(word)
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1
    return vocabulary
                
            
    

def nn_graph(vocabulary_size, embeddings):
    """Definición del grafo de la red neuronal.
    
    Returns:
        Las dos entradas del grafo (tensorflow placeholders)
    """

    #Utilizamos los embeddings para inicializar W
    
    W = tf.Variable(tf.constant(0.0, shape=[vocabulary_size, EMBEDDINGS_DIMENSIONS]), trainable=False, name="W")
    
    embedding_placeholder = tf.placeholder(tf.float32, [vocabulary_size, EMBEDDINGS_DIMENSIONS])
    embedding_init = W.assign(embedding_placeholder)

    
    x_sentences_placeholder = tf.placeholder(tf.int32, shape=[None, MAX_LENGTH], name="x_sentences_placeholder")
    y_labels_placeholder = tf.placeholder(tf.int32, shape=[None], name="y_labels_placeholder")
    
    #Capa de embeddings. Aquí generamos los embeddgins de manera aleatoria. En el trabajo 
    #se utilizarán unos embeddings pre-entrenados.
    #word_embeddings = tf.get_variable("word_embeddings", shape=[vocabulary_size, EMBEDDINGS_DIMENSIONS], dtype=tf.float32, trainable=True)
    x_sentences_embeddings = tf.nn.embedding_lookup(W, x_sentences_placeholder, name="layer_embeddings_lookup")
    
    #CNN
    x_sentences_conv_activation = None
    with tf.variable_scope("cnn_layer") as scope:
        #CNN1
        v_kernel = tf.get_variable("kernel", shape=[2,EMBEDDINGS_DIMENSIONS, CNN_OUTPUT_FEATURES], dtype=tf.float32)
        x_sentences_conv = tf.nn.conv1d(x_sentences_embeddings, v_kernel, 1, padding="VALID", name="cnn_operation")
        v_bias = tf.get_variable("bias", shape=[CNN_OUTPUT_FEATURES], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(x_sentences_conv, v_bias)
        x_sentences_conv_activation = tf.nn.tanh(pre_activation, name="cnn_activation")
        #print(x_sentences_conv_activation.get_shape())
        x_sentences_conv_activation = tf.expand_dims(x_sentences_conv_activation,axis=1)
        #print(x_sentences_conv_activation.get_shape())
        x_sentences_conv_activation=tf.nn.max_pool(x_sentences_conv_activation,ksize=[1,1,2,1],strides=[1,1,2,1], padding="VALID", name="cnn_pooling")
        #print(x_sentences_conv_activation.get_shape())
        x_sentences_conv_activation=tf.squeeze(x_sentences_conv_activation,axis=[1])
        #print(x_sentences_conv_activation.get_shape())

        #CNN2
        v_kernel_2 = tf.get_variable("kernel_2", shape=[3,CNN_OUTPUT_FEATURES, CNN2_OUTPUT_FEATURES], dtype=tf.float32)
        x_sentences_conv2 = tf.nn.conv1d(x_sentences_conv_activation, v_kernel_2, 1, padding="VALID", name="cnn_operation_2")
        v_bias_2 = tf.get_variable("bias_2", shape=[CNN2_OUTPUT_FEATURES], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        pre_activation_2 = tf.nn.bias_add(x_sentences_conv2, v_bias_2)
        x_sentences_conv_activation_2 = tf.nn.tanh(pre_activation_2, name="cnn_activation_2")
        #print(x_sentences_conv_activation.get_shape())
        x_sentences_conv_activation_2 = tf.expand_dims(x_sentences_conv_activation_2,axis=1)
        #print(x_sentences_conv_activation.get_shape())
        x_sentences_conv_activation_2 =tf.nn.max_pool(x_sentences_conv_activation_2,ksize=[1,1,2,1],strides=[1,1,2,1], padding="VALID", name="cnn_pooling_2")
        #print(x_sentences_conv_activation.get_shape())
        x_sentences_conv_activation_2 =tf.squeeze(x_sentences_conv_activation_2,axis=[1])
        #print(x_sentences_conv_activation.get_shape())
    
    #Pasar de 3 dimensiones [batch, sentence, CNN_OUTPUT_FEATURES] a [batch, sentence*CNN_OUTPUT_FEATURES]
    x_sentences_conv_activation_new_shape = [-1, x_sentences_conv_activation_2.get_shape()[1] * x_sentences_conv_activation_2.get_shape()[2]]
    x_sentences_conv_activation = tf.reshape(x_sentences_conv_activation_2,x_sentences_conv_activation_new_shape, name="reshaping")
    #Full connect layer
    x_sentences_dense_activation = None
    with tf.variable_scope("dense_layer") as scope:
        
        weights = tf.get_variable("dense_weigths", shape=[x_sentences_conv_activation.get_shape()[1].value, x_sentences_conv_activation.get_shape()[1].value], dtype=tf.float32)
        bias = tf.get_variable("dense_variables", shape=[x_sentences_conv_activation.get_shape()[1].value], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        x_sentences_dense_activation = tf.tanh(tf.matmul(x_sentences_conv_activation, weights) + bias, name="dense_layer")
        
    #Softmax layer
    y_classified = None
    with tf.variable_scope("softmax_layer") as scope:
        #2 es el número de clases.
        weights = tf.get_variable("softmax_weights", shape=[x_sentences_dense_activation.get_shape()[1].value,2])
        bias = tf.get_variable("softmas_bias", shape=[2], initializer=tf.constant_initializer(0.1))
        y_logits = tf.matmul(x_sentences_dense_activation, weights) + bias
        y_classified = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_labels_placeholder, logits=y_logits, name="softmax")
        
    #Loss function
    f_loss = tf.reduce_mean(y_classified, name="f_loss")
    train_step = tf.train.AdadeltaOptimizer().minimize(f_loss, name="nn_train_step")

    accuracy = None
    with tf.variable_scope("accuracy"):
        prediction_labels = tf.argmax(y_logits, 1, name="prediction_labels")
        correct_predictions = tf.equal(prediction_labels, tf.to_int64(y_labels_placeholder), name="correct_predicionts")
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
        
    return x_sentences_placeholder, y_labels_placeholder


def get_features(training_sentences, vocabulary):
    """Get the features of the batch_sentences.
    
    Args:
        batch_sentences: Lista de lista de oraciones y palabras.
        vocabulary: Diccionario con las palabras de entrenameinto y su identificador o índice.
    Returns:
        Lista de listas de enteros
     
    """
    sentences_features = []
    own_lower = str.lower
    for sent in training_sentences:
        sentence_features = [vocabulary.get(own_lower(word), OOV_INDEX) for word in sent]
        sentences_features.append(sentence_features)
    return sentences_features


def padding_truncate(training_sentences):
    """Amplia o recorta las oraciones de entrenamiento.
    
    Args:
        training_sentences: Lista de listas de enteros.
    """
    
    for i in range(len(training_sentences)):
        sent_size = len(training_sentences[i])
        if sent_size > MAX_LENGTH:
            training_sentences[i] = training_sentences[i][:MAX_LENGTH]
        elif sent_size < MAX_LENGTH:
            training_sentences[i] += [0] * (MAX_LENGTH - sent_size)
    
    return training_sentences 
            

def model_training(training_sents, training_labels, vocabulary, 
                   x_sentences_placeholder, y_labels_placeholder):
    """Entrenamiento del modelo.
    
    Args:
        training_corpus: lista de lista de oraciones
        training_labels:  lista de enteros que se corresponden con las etiquetas
        
    Returns:
        El modelo (NN) entrenado.
    """
    #Preparación de la entrada: cálculo de características
    training_sents_features = get_features(training_sents, vocabulary)
    #Preparación de la entrada: Ampliación (padding) o truncado (truncate)
    training_sents_features = padding_truncate(training_sents_features)
    
    nn_model = tf.Session()
    
    #Es muy importante este paso, dado que inicializa todas las variables
    nn_model.run(tf.initialize_all_variables())
    
    number_of_batches = int(len(training_sents_features)/SIZE_BATCHES)
    
    
    #Esto son los nombres de los nodos que queremos ejecutar del grafo de la NN.
    #Mirar ref. de session.run(): https://www.tensorflow.org/api_docs/python/tf/Session#run
    nn_fetches = {"nn_train_step":"nn_train_step",
                  "accuracy":"accuracy/accuracy:0",
                  "f_loss":"f_loss:0"}
    
    for epoch in range(EPOCHS):
        accuracy_batch_values = []
        f_loss_batch_values = []
        for batch in range(number_of_batches):
             
             start_index = batch * SIZE_BATCHES
             end_index = (batch + 1) * SIZE_BATCHES
             batch_sentences = training_sents_features[start_index:end_index]
             batch_labels = training_labels[start_index:end_index]
             n_batch_sentences = len(batch_sentences)
             if n_batch_sentences != 0: #Si el tamaño del batch es cero, no se hace nada.
                 #Si el nº. de oracione en el batch es menor que el tamaño del batch, rellenamos con las del principio del corpus de entrenamiento.
                 if n_batch_sentences < SIZE_BATCHES:
                     batch_sentences += training_sents_features[0:(SIZE_BATCHES - n_batch_sentences)]
                     batch_labels += training_labels[0:(SIZE_BATCHES - n_batch_sentences)]
                 dict_to_feed = {x_sentences_placeholder:batch_sentences,
                                 y_labels_placeholder:batch_labels}
                 train_fetch_values = nn_model.run(nn_fetches, feed_dict=dict_to_feed)
                 str_to_print = "Epoch: {} Batch: {}: Loss: {} Accuracy: {}".format(epoch+1,batch+1,train_fetch_values["f_loss"],train_fetch_values["accuracy"])
                 print(str_to_print)
                 
                 accuracy_batch_values.append(float(train_fetch_values["accuracy"]))
                 f_loss_batch_values.append(float(train_fetch_values["f_loss"]))
        str_to_print = "Epoch: {} Mean loss: {} Mean accuracy: {}".format(epoch+1,statistics.mean(f_loss_batch_values),statistics.mean(accuracy_batch_values))
        print(str_to_print)
    return nn_model
                 
def model_evaluation(nn_model, test_sentences, test_labels, vocabulary, 
                     x_sentences_placeholder, y_labels_placeholder):
    """Evalúa el modelo
    
    Args:
        nn_model: el modelo correspondiente a la NN.
        test_sentences: Lista de listas con las oraciones del test
        test_labels: lista con las clases de los ejemplos de test
        x_sentences_placeholder: Entrada de la NN correspondiente a las oraciones
        y_labels_placeholders: Entrada de la NN correspondiente a las etiquetas
    """
    
    #Preparación de la entrada: cálculo de características
    test_sents_features = get_features(test_sentences, vocabulary)
    #Preparación de la entrada: Ampliación (padding) o truncado (truncate)
    test_sents_features = padding_truncate(test_sents_features)
    
    #Definición de las operaciones que hay que ejecutar en el test. En este
    #caso no se ejecuta el "training_step" porque no queremos entrenar.
    test_fetches = {"accuracy":"accuracy/accuracy:0",
                    "f_loss":"f_loss:0"}
    
    dict_fed = {x_sentences_placeholder:test_sents_features,
                 y_labels_placeholder:test_labels}
    test_fetches_values = nn_model.run(test_fetches, feed_dict=dict_fed)
    
    str_to_print = "Accuracy: {} Loss: {}".format(test_fetches_values["accuracy"], test_fetches_values["f_loss"])
    print(str_to_print)
    
                

if __name__ == '__main__':


    #Definir semilla aleatoria
    random.seed(RANDOM_SEED)
    print("1.- Random seed: {}".format(RANDOM_SEED))
    
    #2.- Leer corpus y partición de entrenamiento y test.
    print("2.- Preparación partición de datos.")
    train_sents, test_sents, train_labels, test_labels = data_preparation_train_test()
    print("Total: {}\nEntrenamiento: {}\nTest: {}\n----".format(len(train_sents)+len(test_sents), len(train_sents), len(test_sents)))
   
    #3.- Creación del vocabulario de entrenamiento. Toda palabra que no esté en
    #el vocabulario de entrenamiento se consdierá palabra fuera de vocabulario (00).
    #Si tratamos de asimilarlo a otro problema de aprendizaje automática, las OOV
    #serían datos perdidos.
    
    #¿Por qué se define el inicio de índice en 2? Por que se suele reservar el
    #índice 0 para las palabras 00V, y el índice 1 para el padding (extensión de la entrada de la red)..
    #print("3.- Construcción del vocabulario (solo entrenamiento)")    
    print("3.- Construcción del vocabulario")    
    
    #train_vocabulary = build_vocabulary(train_sents, 2)

    vocab,embd = loadAmazonWE('../AmazonWE/sentic2vec.csv')
    vocab_size = len(vocab)
    embedding = np.asarray(embd)

    train_vocabulary = build_vocabulary(vocab, 2)


    print("Tamaño voc.: {}\n---".format(len(train_vocabulary)))
    
    #4.- Compilamos el grafo.
    print("4.- Compilación del grafo de la red neuronal\n---")
    x_sentences_placeholder, y_labels_placeholder = nn_graph(vocab_size, embedding)
    
    #5.- Entrenamiento
    print("5.- Entrenamiento del modelo")
    nn_model = model_training(train_sents, train_labels, train_vocabulary, 
                              x_sentences_placeholder, y_labels_placeholder)
    
    print("-- Fin Entrenamiento --")
    #6.- Evaluación
    print("6.- Evaluación del modelo")
    model_evaluation(nn_model, test_sents, test_labels, train_vocabulary, 
                     x_sentences_placeholder, y_labels_placeholder)
    
    print("-- Fin Evaluación --")
    


    
    
    