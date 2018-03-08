from __future__ import print_function

from random import random
from numpy import array
from numpy import cumsum
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import LSTM, Embedding
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional, Concatenate, concatenate, Dropout, Input, Merge, TimeDistributed, Flatten
from keras.models import Model
import data_set
from data_set import load_embeddings, fetch_data, fetch_embeddings
from data_set import MAX_SEQUENCE_LENGTH_HEAD
from data_set import MAX_SEQUENCE_LENGTH_BODY
from data_set import EMBEDDING_DIM
from data_set import VALIDATION_SPLIT
from data_set import MAX_NUM_WORDS, embeddings_index, labels_index
import numpy as np
from keras.utils import to_categorical
import pickle
from sklearn.metrics import accuracy_score
import tensorflow as tf

load_embeddings(BASE_DIR = '/home/harsha/stance-detection')
(headlines, bodies_map, labels) = fetch_data(BASE_DIR = '/home/harsha/stance-detection')

(data_head, data_body, embedding_layer_head, embedding_layer_body) = fetch_embeddings(headlines, bodies_map)
pickle.dump((headlines, bodies_map, labels), open("train_file1.txt", "wb"))
pickle.dump((data_head, data_body, embedding_layer_head, embedding_layer_body), open("train_file2.txt", "wb"))
(headlines, bodies_map, _) = pickle.load(open("train_file1.txt", "rb"))
(data_head, data_body, embedding_layer_head, embedding_layer_body) = pickle.load(open("train_file2.txt", "rb"))

numLSTM = 200
outputClasses = 4
inputTitle = Input(shape=(MAX_SEQUENCE_LENGTH_HEAD, ), dtype='int32', name='input_title')
inputBody = Input(shape=(MAX_SEQUENCE_LENGTH_BODY, ), dtype='int32', name='input_body')

embedded_sequence_head = embedding_layer_head(inputTitle)
embedded_sequence_body = embedding_layer_body(inputBody)

titleLSTM = Bidirectional(LSTM(numLSTM, return_sequences = False))
bodyLSTM = Bidirectional(LSTM(numLSTM, return_sequences = False))

output1 = titleLSTM(embedded_sequence_head)
output2 = bodyLSTM(embedded_sequence_body)

concatTitleBody = Merge(mode="concat",name="concat_layer")([output1, output2])

hiddenDense = Dense(100, activation = 'relu')
dropout = Dropout(0.3)

outputDense = Dense(outputClasses, activation = 'softmax', name = 'out')

concat = hiddenDense(concatTitleBody)
concat = dropout(concat)
out = outputDense(concat)
# out.get_config()
# X = np.zeros(10, )

model = Model(inputs = [inputTitle, inputBody], outputs = [out])
model.summary()

import keras.backend as K
def w_categorical_crossentropy(y_true, y_pred):
#     print(K.eval(tf.shape(y_true)), tf.shape(y_pred))
#     print(y_true, y_pred)
    p0 = tf.map_fn(lambda x:x[0], y_pred, dtype=tf.float32)
    import math
#     print(p0)
    loss = -tf.log(p0)
    param = 1
    sumP = y_pred[:,1] + y_pred[:,2] + y_pred[:,3]
    for i in range(1,4):
        loss += (-(y_true[:,i]*tf.log(1-p0)) - param*(y_true[:,i]*tf.log(y_pred[:,i]/())))
    return loss