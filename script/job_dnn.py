import pandas as pd 
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler
import numpy as np 
from sklearn.cross_validation import train_test_split
import cPickle,re
import theano
import theano.tensor as T
from nolearn.lasagne import BatchIterator, NeuralNet
from lasagne.objectives import aggregate, squared_error, categorical_crossentropy
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer,Conv1DLayer, Conv2DLayer, MaxPool2DLayer
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax, sigmoid
from nolearn.lasagne import TrainSplit
from prepro import * 
from cnn_job2 import *
from lasagne import nonlinearities
from autoencoder import  AutoEncoder



def float32(k):
    return np.cast['float32'](k)


def mlogloss(y, t, eps=1e-15):
    """
    cross entropy loss, summed over classes, mean over batches
    modified with ordered class , eg, size 1 to 7.
    """
    y = T.clip(y, eps, 1 - eps) *  T.log(np.array([10,20,30,40,50,60,70]))
    loss = -T.sum(t * T.log(y)) / y.shape[0].astype(theano.config.floatX)
    return loss

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None
    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class AdjustDropout(object):
    def __init__(self, name, start=0.2, stop=0.1):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None
    def __call__(self, nn, train_history):
        if self.ls is None:
            #self.ls = np.repeat(np.linspace(self.stop, self.start, 4), int(nn.max_epochs/4))
            #self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
            self.ls = np.repeat(np.array([[self.start, self.stop]]), int(nn.max_epochs/2))
        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

def train_dnn(train, train_y, test):
    num_features = train.shape[1]
    num_classes = len(list(set(train_y)))
    layers0 = [('input', InputLayer),
     	      ('dropout0', DropoutLayer),
               ('dense0', DenseLayer), 
               ('dropout1', DropoutLayer),
    	   ('dense1', DenseLayer),
    	   ('dropout2', DropoutLayer),
               ('output', DenseLayer)]


    net0 = NeuralNet(layers=layers0,
                     input_shape=(None, num_features),
                     dropout0_p = 0.1, #theano.shared(float32(0.1)),
    		 dense0_num_units= 5000,
                     dropout1_p= 0.3, #theano.shared(float32(0.5)),
    	         dense1_num_units = 10000,
    		 dropout2_p = 0.5,
                     output_num_units=num_classes,
                     output_nonlinearity=softmax,
                     update=nesterov_momentum,
                     #update_learning_rate=0.003,
                     #update_momentum=0.9,
                     update_learning_rate = theano.shared(float32(0.001)),
        		  update_momentum=theano.shared(float32(0.9)),
                     objective_loss_function = categorical_crossentropy,
                     train_split = TrainSplit(0.2),
                     verbose=1,
                     max_epochs=150,
    		 on_epoch_finished=[
    		  EarlyStopping(patience = 20),	
    		  AdjustVariable('update_learning_rate', start=0.001, stop=0.0001),
            	  AdjustVariable('update_momentum', start=0.9, stop=0.999),
    		 ]
    )
    net0.fit(train, train_y)
    print('Prediction Complete')
    pred1 = net0.predict_proba(test)
    return pred1

def train_autoencoder(train, train_y, test):
    num_features = train.shape[1]
    net = NeuralNet(
        layers=[
            ('input', InputLayer),
            ('auto', AutoEncoder),
            ('output', DenseLayer),
            ],
        input_shape=(None, 1, num_features),
        auto_num_units = 1000, 
        auto_n_hidden = 10,
        output_num_units=1000, 
        update_learning_rate=theano.shared(float32(0.03)),
        update_momentum=theano.shared(float32(0.9)),
        output_nonlinearity=nonlinearities.softmax,
        regression=True,
        max_epochs=3,
        verbose=1,
    )
    net.fit(train, train_y)
    with open('net.pickle', 'wb') as f:
        pickle.dump(net, f, -1)
    pred_auto = net.predict_proba(test)
    return pred_auto

def load_add_feature():
    print('--- load data ')
    offset =30000
    train, test = load_data()
    #---------------------- filtering out the non-32 positions
    train['position_name'] = train['workExperienceList'].apply(lambda x : x[1]['position_name'])
    train = train.loc[train['position_name'].isin(jobs)]
    names = train.columns
    train = getFeature(train)
    test = getFeature(test)
    train, test = getFeatureTotal(train, test)
    le = LabelEncoder()
    train['position_name'] = le.fit_transform(train['position_name'])
    #train['salary'] = train['workExperienceList'].apply(lambda x :  x[1]['salary'])
    #le = LabelEncoder()
    #train['salary'] = le.fit_transform(train['salary'])
    train, val , train_y, val_y = train[:offset], train[offset:], train['position_name'][:offset], train['position_name'][offset:]
    train = train.drop(['id'], 1)
    val = val.drop(['id'], 1)
    test = test.drop(['id'], 1)
    print('generate log features of position_name')
    feature_list = ['major', 'gender', 'age', 'last_dep', 'last_pos','last_industry', \
    'pre_pos', 'last_salary', 'last_size' , 'pre2_industry', 'pre2_pos' ]
    train, val, test = getMultiLogFeatures(train, val, test, feature_list, 'position_name')
    train = train.drop(['degree', 'position_name'],1)
    val = val.drop(['degree', 'position_name'],1)    
    test = test.drop(['degree'], 1)
    train = train.drop(['workExperienceList'], 1)
    val = val.drop(['workExperienceList'], 1)
    test = test.drop(['workExperienceList'], 1)
    sle = StandardScaler()
    sle.fit(train)
    train = sle.transform(train)
    test = sle.transform(test)
    train = np.array(train, dtype = np.float32)
    test = np.array(test,  dtype = np.float32)
    train_y = np.array(train_y, dtype = np.int32)
    return train, train_y, test , le

train, train_y , test, le = load_add_feature()
pred0 = train_autoencoder(train, train_y, test)

pred1 = train_dnn(train, train_y, test)
pred2 = preprocess_pos()
pred3, pred4 = train_cnn_lstm()
pred = np.argmax(pred1  + pred2 + pred3 + pred4 , axis = 1)
pred = [int(x) for x in pred]
sub = pd.read_csv('result/benchmark.csv')
sub['position_name'] = le.inverse_transform(pred)#pd.Series([reverse_dict[x] for x in pred])
#sub['position_name'] = pd.Series([int(x)  for x in pred])
#sub['size'] = pd.Series([int(int(x) /10) for x in pred])
sub.to_csv('result/ensemble_pos_xgb_nn_cnn_lstm_cnnlstm.csv' , index=False)
# cols = sub.columns.values.tolist()[1:]
# sub[cols] = pd.DataFrame(np.around(pred, decimals=5)).applymap(lambda x: round(x, 5))
# sub.to_csv('benchmark_nn_layer1.csv', index=False)
