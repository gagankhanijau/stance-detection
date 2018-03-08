from keras.layers import LSTM, Embedding
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np

import os
import sys
import csv

labels_index = {}
labels_index['unrelated'] = 0
labels_index['agree'] = 1
labels_index['disagree'] = 2
labels_index['discuss'] = 3

BASE_DIR = 'C:/Users/rachu/Google Drive/UCI/q2/NLP/project'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join('', 'fnc-1')
MAX_SEQUENCE_LENGTH_HEAD = 50
MAX_SEQUENCE_LENGTH_BODY = 200
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
MAX_NUM_WORDS = 20000

embeddings_index = {}

def load_embeddings(BASE_DIR = 'C:/Users/rachu/Google Drive/UCI/q2/NLP/project'):
    with open(os.path.join(BASE_DIR, 'glove.6B.50d.txt'), encoding = 'utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))



def fetch_data(BASE_DIR = 'C:/Users/rachu/Google Drive/UCI/q2/NLP/project/data', mode = 'train'):
    headlines = []
    labels = []
    bodies_map = {}

    if len(embeddings_index) == 0:
        load_embeddings()
    bodyCSV = 'train_bodies.csv'
    stanceCSV = 'train_stances.csv'

    if mode == 'test':
        bodyCSV = 'competition_test_bodies.csv'
        stanceCSV = 'competition_test_stances.csv'

    with open(os.path.join(BASE_DIR, bodyCSV), 'r', encoding = 'utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            bid, text = line
            bodies_map[bid] = text
            
    with open(os.path.join(BASE_DIR, stanceCSV), encoding = 'utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            if len(line) == 2:
                hl, bid = line
                stance = 'unrelated'
            else:
                hl, bid, stance = line

            headlines.append((hl, bid, labels_index[stance]))
    print('All done')
    return (headlines, bodies_map, labels)



def fetch_embeddings(headlines, bodies_map):
    tokenizer_headline = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer_headline.fit_on_texts(list(zip(*headlines))[0])
    sequences_head = tokenizer_headline.texts_to_sequences(list(zip(*headlines))[0])

    word_index_head = tokenizer_headline.word_index

    bodies_indices = list(zip(*headlines))[1]
    bodies = []

    for i in bodies_indices:
        bodies.append(bodies_map[i])

    tokenizer_body = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer_body.fit_on_texts(bodies)
    sequences_body = tokenizer_body.texts_to_sequences(bodies)
    word_index_body = tokenizer_body.word_index

    labels = to_categorical(np.asarray(list(zip(*headlines))[2]))

    data_head = pad_sequences(sequences_head, maxlen=MAX_SEQUENCE_LENGTH_HEAD)
    data_body = pad_sequences(sequences_body, maxlen=MAX_SEQUENCE_LENGTH_BODY)

    print('Preparing embedding matrix.')

    num_words_head = min(MAX_NUM_WORDS, len(word_index_head) + 1)
    embedding_matrix_head = np.zeros((num_words_head, EMBEDDING_DIM))
    for word, i in word_index_head.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix_head[i] = embedding_vector

    num_words_body = min(MAX_NUM_WORDS, len(word_index_body) + 1)
    embedding_matrix_body = np.zeros((num_words_body, EMBEDDING_DIM))
    for word, i in word_index_body.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix_body[i] = embedding_vector

    print('All done')
    embedding_layer_head = Embedding(num_words_head,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix_head],
                                input_length=MAX_SEQUENCE_LENGTH_HEAD,
                                trainable=False)

    embedding_layer_body = Embedding(num_words_body,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix_body],
                                input_length=MAX_SEQUENCE_LENGTH_BODY,
                                trainable=False)
    return (data_head, data_body, embedding_layer_head, embedding_layer_body)
    # return (data_head, data_body, embedding_matrix_head, embedding_matrix_body)

