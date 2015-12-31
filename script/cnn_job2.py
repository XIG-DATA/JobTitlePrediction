# coding: utf-8
from __future__ import absolute_import
from os import path
import os
import re
import codecs
import pandas as pd, json
import numpy as np
from lasagne import layers
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from pandas import DataFrame
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import theano
from lasagne import nonlinearities
from autoencoder import  AutoEncoder

def load_txt(x):
    with open(x) as f:
        res = [t.decode('gbk','ignore') for t in f]
        return ''.join(res)

import jieba
def cutword_1(x):
    words = jieba.cut(x, cut_all = False)
    return ' '.join(words)

def getNewsData():
    rootdir = '../Reduced/'
    dirs = os.listdir(rootdir)
    dirs = [path.join(rootdir,f) for f in dirs if f.startswith('C')]
    text_t = {}
    for i, d in enumerate(dirs):
        files = os.listdir(d)
        files = [path.join(d, x) for x in files if x.endswith('txt') and not x.startswith('.')]
        text_t[i] = [load_txt(f) for f in files]
    flen = [len(t) for t in text_t.values()]
    labels = np.repeat(text_t.keys(),flen)
    import itertools
    merged = list(itertools.chain.from_iterable(text_t.values()))
    df = pd.DataFrame({'label': labels, 'txt': merged})
    df['ready_seg'] =df['txt'].str.replace(ur'\W+', ' ',flags=re.U) 
    df['ready_seg'] =df['ready_seg'].str.replace(r'[A-Za-z]+', ' ENG ')
    df['ready_seg'] =df['ready_seg'].str.replace(r'\d+', ' NUM ')
    df['seg_word'] = df.ready_seg.map(cutword_1)
    textraw = df.seg_word.values.tolist()
    textraw = [line.encode('utf-8') for line in textraw]
    y = df.label.values 
    return textraw, y
def getJobData():
    train_list = []
    for line in open('../data/train_clean.json', 'r'):
        train_list.append(json.loads(line))
    train = pd.DataFrame(train_list)
    train = train.fillna(-1)
    train['tind'] = train['workExperienceList'].apply(lambda x : [x[i]['industry'] for i in range(len(x)) if x[i]['industry'] is not None and i !=1])
    train['tdep'] = train['workExperienceList'].apply(lambda x : [x[i]['department'] for i in range(len(x)) if x[i]['department'] is not None and i !=1])
    train['tpos'] = train['workExperienceList'].apply(lambda x : [x[i]['position_name'] for i in range(len(x)) if x[i]['position_name'] is not None and i !=1])
    train['major'] = train['major'].fillna(-1)
    train['gender'] = train['gender'].fillna(-1)
    txts = zip(train.tind.values.tolist(),  train.tpos.tolist(), train.major.tolist() )
    txts = [ x+ z + [m]  for x, z, m in txts]
    txts = [ ' '.join([x for x in y if x is not None and  not isinstance(x, int)])  for y in txts]
    labels = train['workExperienceList'].apply(lambda x : x[1]['position_name'] )
    traindf = pd.DataFrame({'label' : labels.tolist(), 'txt': txts})
    traindf['txt'] = traindf.txt.map(cutword_1)

    test_list = []
    for line in open('../data/test_clean.json', 'r'):
        test_list.append(json.loads(line))
    test = pd.DataFrame(test_list)
    test = test.fillna(-1)
    test['tind'] = test['workExperienceList'].apply(lambda x : [x[i]['industry'] for i in range(len(x)) if x[i] is not None and i !=1])
    test['tdep'] = test['workExperienceList'].apply(lambda x : [x[i]['department'] for i in range(len(x)) if x[i] is not None and i !=1])
    test['tpos'] = test['workExperienceList'].apply(lambda x : [x[i]['position_name'] for i in range(len(x)) if x[i] is not None and i !=1])
    test['major'] = test['major'].fillna(u'无')
    test['gender'] = test['gender'].fillna(u'男')
    test_txts = zip(test.tind.values.tolist(), test.tdep.values.tolist(), test.tpos.tolist(), test.major.tolist() )
    test_txts = [ x+y + z + [m]  for x, y, z, m in test_txts]
    test_txts = [ ' '.join([x for x in y if x is not None and  not isinstance(x, int)]) for y in test_txts]
    testdf = pd.DataFrame({'txt': test_txts})
    testdf['txt'] = testdf.txt.map(cutword_1)
    return traindf, testdf

def train_cnn_lstm():
    traindf, testdf = getJobData()
    jobs = codecs.open('../data/pos_name.txt', 'rb', encoding = 'utf-8').readlines()
    jobs = [ job.strip() for job in jobs]
    traindf = traindf.loc[traindf['label'].isin(jobs)]
    trainraw = traindf.txt.values.tolist()
    trainraw = [line.encode('utf-8') for line in trainraw]
    testraw = testdf.txt.values.tolist()
    testraw =  [line.encode('utf-8') for line in testraw]

    maxfeatures = 50000
    from keras.preprocessing.text import Tokenizer
    token = Tokenizer(nb_words=maxfeatures)
    token.fit_on_texts(trainraw + testraw) 
    train_seq = token.texts_to_sequences(trainraw)
    test_seq = token.texts_to_sequences(testraw)

    np.median([len(x) for x in train_seq]) 
    from sklearn.preprocessing import LabelEncoder
    le= LabelEncoder()
    y = le.fit_transform(traindf['label'])

    nb_classes =32
    from sklearn.cross_validation import train_test_split
    train_X, val_X, train_y, val_y = train_test_split(train_seq, y , train_size=0.8, random_state=1)
    test_X = np.array(test_seq)


    from keras.optimizers import RMSprop
    from keras.preprocessing import sequence
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.embeddings import Embedding
    from keras.layers.convolutional import Convolution1D, MaxPooling1D
    from keras.layers.recurrent  import SimpleRNN, GRU, LSTM
    from keras.callbacks import EarlyStopping
    from keras.utils import np_utils


    maxlen = 80 
    batch_size = 16 
    word_dim = 50 
    nb_filter = 100  
    filter_length = 10 
    hidden_dims = 100 
    nb_epoch = 10     
    pool_length = 40 

    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(train_X, maxlen=maxlen,padding='post', truncating='post')
    X_val = sequence.pad_sequences(val_X, maxlen=maxlen,padding='post', truncating='post')
    X_test = sequence.pad_sequences(test_X, maxlen=maxlen,padding='post', truncating='post')
    print('X_train shape:', X_train.shape)
    print('X_val shape:', X_val.shape)

    Y_train = np_utils.to_categorical(train_y, nb_classes)
    Y_val = np_utils.to_categorical(val_y, nb_classes)


    print('Build model...')
    model = Sequential()

    model.add(Embedding(maxfeatures, word_dim,input_length=maxlen)) 
    model.add(Dropout(0.2))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode="valid",
                            activation="relu"))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(Flatten())
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    result = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
                validation_split=0.1, show_accuracy=True,callbacks=[earlystop])


    score = earlystop.model.evaluate(X_val, Y_val, batch_size=batch_size)
    print('val score:', score)
    classes = earlystop.model.predict_classes(X_val, batch_size=batch_size)
    acc = np_utils.accuracy(classes, val_y) # 要用没有转换前的y
    print('Validation accuracy:', acc)
    pred_cnn = earlystop.model.predict(X_test, batch_size = batch_size)

    print('2 LSTM...')
    model = Sequential()
    model.add(Embedding(maxfeatures, word_dim,input_length=maxlen)) 
    #model.add(Dropout(0.25))
    model.add(LSTM(100)) 
    model.add(Flatten())
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    result = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
                validation_split=0.1, show_accuracy=True,callbacks=[earlystop])

    score = earlystop.model.evaluate(X_val, Y_val, batch_size=batch_size)
    print('val score:', score)
    classes = earlystop.model.predict_classes(X_val, batch_size=batch_size)
    acc = np_utils.accuracy(classes, val_y) 
    print('Validation accuracy:', acc)
    pred_lstm = earlystop.model.predict(X_test, batch_size = batch_size)

    print('3 CNN + LSTM model...')
    model = Sequential()

    model.add(Embedding(maxfeatures, word_dim,input_length=maxlen)) 
    model.add(Dropout(0.1))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode="valid",
                            activation="relu"))

    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(100))
    #model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    from keras.models import Graph
    fw = [2,10, 5]
    pool_length = [2,50, 10]
    print('Build model...')
    graph = Graph()
    graph.add_input(name='input', input_shape=(maxlen,), dtype=int)
    graph.add_node(Embedding(maxfeatures, word_dim, input_length=maxlen),
                   name='embedding', input='input')


    graph.add_node(Convolution1D(nb_filter=nb_filter,filter_length=fw[0],
                            activation="relu"),
                   name='conv1', input='embedding') 
    graph.add_node(MaxPooling1D(pool_length =pool_length[0], ignore_border = False), name='pool1', input = 'conv1')
    graph.add_node(Flatten(), name='flat1', input='conv1')



    graph.add_node(Convolution1D(nb_filter=nb_filter,filter_length=fw[1],
                            activation="relu"),
                   name='conv2', input='embedding') 
    graph.add_node(MaxPooling1D(pool_length =pool_length[1], ignore_border = False), name='pool2', input = 'conv2')
    graph.add_node(Flatten(), name='flat2', input='conv2')


    graph.add_node(Convolution1D(nb_filter=nb_filter,filter_length=fw[2],
                            activation="relu"),
                   name='conv3', input='embedding') 
    graph.add_node(MaxPooling1D(pool_length =pool_length[2], ignore_border = False), name='pool3', input = 'conv3')
    graph.add_node(Flatten(), name='flat3', input='conv3')

    graph.add_node(Dense(hidden_dims,activation='relu'), name='dense1', 
                   inputs=['flat1', 'flat2', 'flat3'], merge_mode='concat')
    graph.add_node(Dropout(0.4), name='drop1', input='dense1') 
    graph.add_node(Dense(nb_classes, activation='softmax'), name='softmax', input='drop1')
    graph.add_output(name='output', input='softmax')
    graph.compile('Adam', loss = {'output': 'categorical_crossentropy'})

    result = graph.fit({'input':X_train, 'output':Y_train}, 
                       nb_epoch=3,batch_size=batch_size,
                       validation_split=0.1)
    predict = graph.predict({'input':X_test}, batch_size=batch_size)
    pred_cnn_lstm = predict['output']
    classes = pred_cnn_lstm.argmax(axis=1)
    return pred_cnn, pred_lstm, pred_cnn_lstm

