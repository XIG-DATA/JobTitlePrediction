# -*- coding:utf-8 -*-
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler, Imputer
from sklearn.preprocessing import OneHotEncoder
# from nltk.corpus import stopwords
# import nltk
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import codecs
import numpy as np 
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import cPickle,re, os, json
from copy import deepcopy
import jieba
import sys , operator
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
reload(sys)  
sys.setdefaultencoding('utf-8')

jobs_en = codecs.open('../data/pos_class.txt', 'rb').readlines()
jobs_en = [ job.strip().split("(")[0] for job in jobs_en]
jobs = codecs.open('../data/pos_name.txt', 'rb', encoding = 'utf-8').readlines()
jobs = [ job.strip() for job in jobs]
jobs_dict = {}
reverse_dict = {}
jobi = 0
for job in jobs:
    jobs_dict[job] = jobi
    reverse_dict[jobi] = job
    jobi = jobi + 1

indi = 0
ind_dict = {}

inds = codecs.open('ind_dict.txt', 'rb', encoding = 'utf-8').readlines()
for ind in inds:
    ind_dict[ind.split(':')[0]] = indi
    indi = indi + 1

pos_dict = {}
poss = codecs.open('../data/pos_rank.txt', 'rb', encoding = 'utf-8').readlines()
for ind in poss:
    title, rank = ind.strip().split(":")
    pos_dict[title] = rank

wa = 0.35
wb = 0.86
wc = 0.74
wd = 2.25

# model = read_vec("../data/glove.6B.100d.txt")
# print np.mean(np.array([model[part.lower()]  for part in jobs_en[0].strip().split()]) , 0)
# jobvec = np.array([ np.mean(np.array([model[part.lower()] for part in job.strip().split() if part.lower() not in stopwords.words('english') ]), 0 ) for job in jobs_en])
# print jobvec.shape

def get_max_duration_length(x):
    return []

def get_job_duration(x, i):
    try :
        end_year =  int(x[i]['end_date'][:4])
    except : 
        end_year = 2015
    try :
        end_month =  int(x[i]['end_date'][5:])
    except :
        end_month =  12
    try :
        start_year =  int(x[i]['start_date'][:4])
    except :
        start_year = end_year -1 
    try :
        start_month  = int(x[i]['start_date'][5:])
    except :
        start_month = end_month -1
    return (end_year - start_year) * 12 + end_month - start_month

def get_interval(x, i, j):
    try :
        end_year =  int(x[i]['end_date'][:4])
    except:
        end_year = 2015
    try:
        end_month =  int(x[i]['end_date'][5:])
        start_year =  int(x[j]['start_date'][:4])
        start_month  = int(x[j]['start_date'][5:])
        r = (end_year - start_year) * 12 + end_month - start_month
    except :
        r = -1
    return r

def work_total_month(x):
    try :
        end_year =  int(x[0]['end_date'][:4])
    except:
        end_year = 2015
    try:
        end_month =  int(x[0]['end_date'][5:])
        start_year =  int(x[-1]['start_date'][:4])
        start_month  = int(x[-1]['start_date'][5:])
        r = (end_year - start_year) * 12 + end_month - start_month
    except :
        r = -1
    return r

def num_unique_work(x):
    x = [ x[i] for i in range(len(x)) if i!=1 ]
    xlist = [ xx['position_name']  if xx is not None else 'none' for xx in x  ]
    return len(set(xlist))


def benchmark(train):
    pred_size = train['workExperienceList'].apply( lambda x : np.round((x[2]['size'] )) )
    acc_size = np.sum(pred_size == train['workExperienceList'].apply(lambda x: x[1]['size']))/1.0/len(train)
    print('benchmarking acc of size:' + str(acc_size) )
    pred_salary = train['workExperienceList'].apply( lambda x : np.round((x[0]['salary'] )) )
    acc_salary = np.sum(pred_salary == train['workExperienceList'].apply(lambda x: x[1]['salary']))/1.0/len(train)
    print('benchmarking acc of salary:' + str(acc_salary) )
    train = train.loc[train['workExperienceList'].apply(lambda x: x[1]['position_name']).isin(jobs)]
    pred_pos = train['workExperienceList'].apply( lambda x : x[0]['position_name']  )
    acc_pos = np.sum(pred_pos == train['workExperienceList'].apply(lambda x: x[1]['position_name'] if x[1]['position_name'] in jobs else u"销售经理" ))/1.0/len(train)
    print('benchmarking acc of pos: ' + str(acc_pos) )
    rough_score(0.7, acc_size, acc_salary, acc_pos)

def rough_score(acc_deg, acc_size, acc_salary, acc_pos):
    score = (acc_deg * wa + acc_size * wb + acc_salary * wc + acc_pos * wd)/(wa+wb+wc+wd)
    print('rough estimation of final score: ' + str(score))

def auto_transform(train, test, nameA):
    le = LabelEncoder()
    #train[nameA] = train[nameA].apply(lambda x :  " ".join(jieba.cut(x)).split()[0] if x is not None and len(" ".join(jieba.cut(x)).split()) > 0 else 'none')
    #test[nameA]  = test[nameA].apply(lambda x :  " ".join(jieba.cut(x)).split()[0] if x is not None  and len(" ".join(jieba.cut(x)).split()) > 0 else 'none')
    le.fit(list(train[nameA]) + list(test[nameA]))
    train[nameA] = le.transform(train[nameA])
    test[nameA]  = le.transform(test[nameA])
    return train, test

def getFeatureTotal(train, test):
    le = LabelEncoder()
    le.fit(list(test['last_pos']) + list(train['last_pos'])) 
    train['last_pos'] = le.transform(train['last_pos'])
    test['last_pos'] = le.transform(test['last_pos'])
    le = LabelEncoder()
    train['last_industry'] = train['last_industry'].apply(lambda x :  " ".join(jieba.cut(x)).split()[0] if x is not None and len(" ".join(jieba.cut(x)).split()) > 0 else 'none')
    test['last_industry']  = test['last_industry'].apply(lambda x :  " ".join(jieba.cut(x)).split()[0] if x is not None  and len(" ".join(jieba.cut(x)).split()) > 0 else 'none')
    le.fit(list(train['last_industry']) + list(test['last_industry']))
    train['last_industry'] = le.transform(train['last_industry'])
    test['last_industry'] = le.transform(test['last_industry'])
    le = LabelEncoder()
    train['last_dep'] = train['last_dep'].apply(lambda x :  " ".join(jieba.cut(x)).split()[0] if x is not None and len(" ".join(jieba.cut(x)).split()) > 0 else 'none')
    test['last_dep']  = test['last_dep'].apply(lambda x :  " ".join(jieba.cut(x)).split()[0] if x is not None  and len(" ".join(jieba.cut(x)).split()) > 0 else 'none')
    le.fit(list(train['last_dep']) + list(test['last_dep']))
    train['last_dep'] = le.transform(train['last_dep'])
    test['last_dep']  = le.transform(test['last_dep'])
    
    le.fit(list(test['pre_pos']) +  list(train['pre_pos']))
    train['pre_pos'] = le.transform(train['pre_pos'])
    test['pre_pos'] = le.transform(test['pre_pos'])
    train['pre_industry'] = train['pre_industry'].apply(lambda x :  " ".join(jieba.cut(x)).split()[0] if x is not None and len(" ".join(jieba.cut(x)).split()) > 0 else 'none')
    test['pre_industry']  = test['pre_industry'].apply(lambda x :  " ".join(jieba.cut(x)).split()[0] if x is not None  and len(" ".join(jieba.cut(x)).split()) > 0 else 'none')
    le.fit(list(train['pre_industry']) + list(test['pre_industry']))
    train['pre_industry'] = le.transform(train['pre_industry'])
    test['pre_industry'] = le.transform(test['pre_industry'])
    train['pre_dep'] = train['pre_dep'].apply(lambda x :  " ".join(jieba.cut(x)).split()[0] if x is not None and len(" ".join(jieba.cut(x)).split()) > 0 else 'none')
    test['pre_dep']  = test['pre_dep'].apply(lambda x :  " ".join(jieba.cut(x)).split()[0] if x is not None  and len(" ".join(jieba.cut(x)).split()) > 0 else 'none')
    le.fit(list(train['pre_dep']) + list(test['pre_dep']))
    train['pre_dep'] = le.transform(train['pre_dep'])
    test['pre_dep']  = le.transform(test['pre_dep'])
      
    # le.fit(list(test['first_pos']) + list(train['first_pos']))
    # train['first_pos'] = le.transform(train['first_pos'])
    # test['first_pos'] = le.transform(test['first_pos'])
    #train, test = auto_transform(train,test, 'first_industry')
    #train, test = auto_transform(train,test, 'first_dep')
    # train, test = auto_transform(train, test, 'pre2_pos') 
    # train, test = auto_transform(train, test, 'pre2_industry') 
    # train, test = auto_transform(train, test, 'pre2_dep') 
    return train, test


def getSizeFeatures(train):
    # train['salary_ratio'] = train['last_salary']/1.0/train['pre_salary']
    # train['size_ratio']  = train['last_size']/1.0/train['pre_size']
    # train['salary_prod'] = train['last_salary'] * train['pre_salary']
    # train['size_prod']   = train['last_size']  * train['pre_size']
    # train['last_end_year'] = train['workExperienceList'].apply(lambda x : 2015 if not x[0]['end_date'][:4].startswith("20") else int(x[0]['end_date'][:4]) )  
    # train['last_end_month'] = train['workExperienceList'].apply(lambda x : 7 if not x[0]['end_date'][:4].startswith("20") else int(x[0]['end_date'][5:]) )  
    # train['last_ss_ratio'] = train['last_salary']/train['last_size']
    # train['last_ss_prod'] = train['last_salary'] * train['last_size']
    # train['pre_ss_ratio'] = train['pre_salary']/train['pre_size']
    # train['pre_ss_prod'] = train['pre_salary'] * train['pre_size']
    # train['start_work_year'] = train['workExperienceList'].apply(lambda x : 0 if x[len(x)-1]['start_date'] is None else int(x[len(x)-1]['start_date'].split('-')[0]) )  
    # train['max_size'] = train['workExperienceList'].apply(lambda x : np.max([ 0 if x[i] is  None else x[i]['size'] for i in range(len(x)) if i != 1]))
    # train['min_size'] = train['workExperienceList'].apply(lambda x : np.min([0 if x[i] is  None else  x[i]['size'] for i in range(len(x)) if i != 1]))
    # train['size_mm_ratio'] = train['max_size']/1.0/train['min_size']   
    # train['pre_job_long']  = train['workExperienceList'].apply(lambda x : get_job_duration(x, 2))
    # train['pre_job_islong']= train['workExperienceList'].apply(lambda x : 1 if get_job_duration(x, 2) > 36 else 0)
    # train['pre_3tuple_prod'] = train['pre_job_long'] * train['pre_ss_prod']
    # train['2job_ratio'] = train['last_job_long']/1.0/train['pre_job_long']
    # train['last_3tuple_prod'] = train['last_job_long'] * train['last_ss_prod']   
    train['last_dep'] = train['workExperienceList'].apply(lambda x : x[0]['department'] if x[0]['department'] is not None else 'none')
    train['pre_dep']  = train['workExperienceList'].apply(lambda x : x[2]['department'] if len(x)>=2 and x[2]['department'] is not None else 'none')
    train['last_dep'] = train['workExperienceList'].apply(lambda x : x[0]['department'] if x[0]['department'] is not None else 'none')
    train['first_dep']  = train['workExperienceList'].apply(lambda x : x[-1]['department'] if x[-1]['department'] is not None else 'none')
    train['last_industry'] = train['workExperienceList'].apply(lambda x : x[0]['industry'])
    train['pre_industry'] = train['workExperienceList'].apply(lambda x : x[2]['industry'] if x[2]['industry'] is not None else 'none')
    train['work_age'] = train['workExperienceList'].apply(lambda x :work_total_month(x))
    train['last_job_long'] = train['workExperienceList'].apply(lambda x : get_interval(x, 0, 0))
    train['pre_job_long'] = train['workExperienceList'].apply(lambda x : get_interval(x, 2, 2))
    train['first_job_long'] = train['workExperienceList'].apply(lambda x : get_interval(x, -1, -1))
    return train 

def getFeature(train):
    train['last_dep'] = train['workExperienceList'].apply(lambda x : x[0]['department'] if x[0]['department'] is not None else 'none')
    train['last_salary'] = train['workExperienceList'].apply(lambda x : x[0]['salary'])
    train['last_size'] = train['workExperienceList'].apply(lambda x : x[0]['size'])
    train['last_industry'] = train['workExperienceList'].apply(lambda x : x[0]['industry'])
    train['last_pos'] = train['workExperienceList'].apply(lambda x : x[0]['position_name'])
    train['pre_pos'] = train['workExperienceList'].apply(lambda x : x[2]['position_name'] if x[2]['position_name'] is not None else 'none')
    train['pre_dep']  = train['workExperienceList'].apply(lambda x : x[2]['department'] if len(x)>=2 and x[2]['department'] is not None else 'none')
    train['pre_salary'] = train['workExperienceList'].apply(lambda x : x[2]['salary'] if x[2]['salary'] is not None else 1)
    train['pre_size'] = train['workExperienceList'].apply(lambda x : x[2]['size'] if x[2]['size'] is not None else 1)
    train['pre_industry'] = train['workExperienceList'].apply(lambda x : x[2]['industry'] if x[2]['industry'] is not None else 'none')
    # train['pre2_pos'] = train['workExperienceList'].apply(lambda x : x[3]['position_name'] if len(x)>3  and  x[3]['position_name'] is not None else 'none')
    # train['pre2_dep']  = train['workExperienceList'].apply(lambda x : x[3]['department'] if len(x)>3 and x[3]['department'] is not None else 'none')
    # train['pre2_salary'] = train['workExperienceList'].apply(lambda x : x[3]['salary'] if len(x)>3 and x[3]['salary'] is not None else 1)
    # train['pre2_size'] = train['workExperienceList'].apply(lambda x : x[3]['size'] if len(x)>3  and x[3]['size'] is not None else 1)
    # train['pre2_industry'] = train['workExperienceList'].apply(lambda x : x[3]['industry'] if len(x)>3  and x[3]['industry'] is not None else 'none')
    #train['first_dep'] = train['workExperienceList'].apply(lambda x : x[-1]['department'] if x[-1]['department'] is not None else 'none')
    #train['first_pos'] = train['workExperienceList'].apply(lambda x : x[-1]['position_name'])
    #train['first_industry'] = train['workExperienceList'].apply(lambda x : x[-1]['industry'])
    #train['first_salary'] = train['workExperienceList'].apply(lambda x : x[-1]['salary'])
    #train['first_size'] = train['workExperienceList'].apply(lambda x : x[-1]['size'])
    train['work_age'] = train['workExperienceList'].apply(lambda x :work_total_month(x))
    train['last_job_long'] = train['workExperienceList'].apply(lambda x : get_interval(x, 0, 0))
    train['pre_job_long'] = train['workExperienceList'].apply(lambda x : get_interval(x, 2, 2))
    #train['first_job_long'] = train['workExperienceList'].apply(lambda x : get_interval(x, -1, -1))
    train['num_times_work'] = train['workExperienceList'].apply(lambda x : len(x)-1)
    train['num_unique_work']= train['workExperienceList'].apply(lambda x : num_unique_work(x))
    train['age'] = train['age'].apply(lambda x : 0 if len(x.encode('ascii','ignore')) == 0 else int(x.encode('ascii', 'ignore')))
    train['start_work_age'] = train['age'] - train['workExperienceList'].apply(lambda x : work_total_month(x)/12.0) 

    return train         

def createOneHotFeature(train, test, features):
    i = 0
    #train_oneh = np.array([])
    for feature in features:
        enc = OneHotEncoder()
        train_f = enc.fit_transform(np.array(train[feature].reshape(len(train),1)))
        test_f  = enc.fit_transform(np.array(test[feature].reshape(len(test),1)))
        if i > 0:
            train_oneh = np.hstack([train_oneh, train_f.toarray()])
            test_oneh = np.hstack([test_oneh, test_f.toarray()])
        else :
            train_oneh = train_f.toarray()
            test_oneh =  test_f.toarray()
        i = i + 1
    return train_oneh, test_oneh

def getMultiLogFeatures(train, val , test, xlist, nameC):
    i = 0
    num_features = len(train.columns)
    for nameA in xlist:
        if i == 0:
            train1, val1, test1 = getNewLogFeatures(train, val, test, nameA, nameC)
        else :
            trainX, valX, testX = getNewLogFeatures(train, val, test, nameA, nameC) 
            keep_list = trainX.columns[num_features:]
            train1  = pd.concat([train1, trainX[keep_list] ], axis =1 )
            val1 = pd.concat([val1, valX[keep_list]], axis =1 )
            test1 = pd.concat([test1, testX[keep_list]], axis = 1)
        i = i + 1
    return train1, val1, test1

def TFIDFeature(train, val,  test, nameA):
    print('--- generating TFIDF features')
    train_desp_list = []
    test_desp_list = []
    val_desp_list =  []

    print test.columns
    for lists in test['workExperienceList']:
        try:
            lines = ' '.join([ re.sub('(|)', '',lists[i][nameA]) for i in range(len(lists)) if i != 1 and lists[i][nameA] is not None ])
            test_desp_list.append(" ".join(jieba.cut(lines, cut_all=True)))
        except TypeError:
            test_desp_list.append('none')
            continue

    for lists in train['workExperienceList']:
        try:
            lines = ' '.join([ re.sub('(|)', '',lists[i][nameA]) for i in range(len(lists)) if i != 1 and lists[i][nameA] is not None])
            train_desp_list.append(" ".join(jieba.cut(lines, cut_all=True)))
        except TypeError:
            train_desp_list.append('none')
            continue

    for lists in val['workExperienceList']:
        try:
            lines = ' '.join([ re.sub('(|)', '',lists[i][nameA]) for i in range(len(lists)) if i != 1  and lists[i][nameA] is not None])
            val_desp_list.append(" ".join(jieba.cut(lines, cut_all=True)))
        except TypeError:
            val_desp_list.append('none')
            continue

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    vect = TfidfVectorizer(max_features= 10000, strip_accents='unicode',  
        analyzer='char',sublinear_tf=1, ngram_range=(2, 7)
    )
    train_desp_vec= vect.fit_transform(train_desp_list)
    test_desp_vec = vect.transform(test_desp_list)
    val_desp_vec  = vect.transform(val_desp_list)
    return train_desp_vec, val_desp_vec, test_desp_vec

def read_vec(filename):
    f = open(filename, 'rb').readlines()
    wordvecs = {}
    for line in f : 
        words = line.strip().split()
        vecs = np.array(words[1:], dtype = np.float32)
        wordvecs[words[0]] = vecs
    return wordvecs


def parse_data(df,logodds,logoddsPA, NameA, NameC):

    feature_list=df.columns.tolist()
    cleanData=df[feature_list]
    cleanData.index=range(len(df))
    print("Creating A features")
    address_features=cleanData[NameA].apply(lambda x: logodds[x])
    address_features.columns=["logodds"+ NameA + NameC + str(x) for x in range(len(address_features.columns))]
    #print("Creating one-hot variables")
    #dummy_ranks_PD = pd.get_dummies(cleanData['Upc'], prefix='U')
    #dummy_ranks_DAY = pd.get_dummies(cleanData["FinelineNumber"], prefix='FN')
    cleanData["logodds" + NameA + NameC ]=cleanData[NameA].apply(lambda x: logoddsPA[x])
    #cleanData=cleanData.drop("Upc",axis=1)
    #cleanData=cleanData.drop("FinelineNumber",axis=1)
    feature_list=cleanData.columns.tolist()
    features = cleanData[feature_list].join(address_features.ix[:,:])
    return features

from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value 
        in column.
        Columns of other types are imputed with mean of column.
        """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

def get_log_count(trainDF, NameA, NameC):
    addresses=sorted(trainDF[NameA].unique())
    categories=sorted(trainDF[NameC].unique())
    C_counts=trainDF.groupby([NameC]).size()
    A_C_counts=trainDF.groupby([NameA, NameC]).size()
    A_counts=trainDF.groupby([NameA]).size()
    logodds={}
    logoddsPA={}
    MIN_CAT_COUNTS=2

    default_logodds=np.log(C_counts/len(trainDF))- np.log(1.0-C_counts/float(len(trainDF)))
    for addr in addresses:
        PA=A_counts[addr]/float(len(trainDF))
        logoddsPA[addr]=np.log(PA)- np.log(1.-PA)
        logodds[addr]=deepcopy(default_logodds)
        for cat in A_C_counts[addr].keys():
            if (A_C_counts[addr][cat]>MIN_CAT_COUNTS) and A_C_counts[addr][cat]<A_counts[addr]:
                PA=A_C_counts[addr][cat]/float(A_counts[addr])
                logodds[addr][categories.index(cat)]=np.log(PA)-np.log(1.0-PA)
        logodds[addr]=pd.Series(logodds[addr])
        logodds[addr].index=range(len(categories))
    return logodds, logoddsPA, default_logodds

def generate_log_features(trainDF, testDF, logodds, logoddsPA, NameA, NameC, default_logodds):
    addresses=sorted(trainDF[NameA].unique())
    A_counts=trainDF.groupby([NameA]).size()
    categories=sorted(trainDF[NameC].unique())
    features = parse_data(trainDF,logodds,logoddsPA, NameA, NameC)
    collist=features.columns.tolist()[2:]
    # scaler = StandardScaler()
    # scaler.fit(features[collist])
    # features[collist]=scaler.transform(features[collist])
    new_addresses=sorted(testDF[NameA].unique())
    new_A_counts=testDF.groupby(NameA).size()
    only_new=set(new_addresses+addresses)-set(addresses)
    only_old=set(new_addresses+addresses)-set(new_addresses)
    in_both=set(new_addresses).intersection(addresses)
    for addr in only_new:
        PA=new_A_counts[addr]/float(len(testDF)+len(trainDF))
        logoddsPA[addr]=np.log(PA)- np.log(1.-PA)
        logodds[addr]=deepcopy(default_logodds)
        logodds[addr].index=range(len(categories))

    for addr in in_both:
        PA=(A_counts[addr]+new_A_counts[addr])/float(len(testDF)+len(trainDF))
        logoddsPA[addr]=np.log(PA)-np.log(1.-PA)
    features_sub =parse_data(testDF,logodds,logoddsPA, NameA, NameC)
    # scaler.fit(features_test)
    #collist=features_sub.columns.tolist()[1:]
    #features_sub[collist]=scaler.transform(features_sub[collist])
    return features, features_sub

def getDataMatrix(vNumber, new_dep, mat, model):
	for i in range(len(vNumber)):
		ind = vNumber[i]
		try :
			words = ' '.join(new_dep[ind])
			words = re.sub('[^a-zA-Z]+', ' ', words)
			words = [ w.lower() for w in words.split() if w not in stopwords.words('english') ]
			vec = [0] * dim
			count = 0
			for w in words:
				if w in model :
					count = count + 1
					mat[i,:] = mat[i,:] + model[w]
			if count == 0 :
				mat[i,:] = np.array([0] * dim)
			else:
				mat[i,:] = mat[i,:]/count
		except TypeError:
			mat[i,:] = np.array([0] * dim)
	return mat

def load_data():
    train_list = []
    for line in open('../data/train_clean.json', 'r'):
        train_list.append(json.loads(line))
    train = pd.DataFrame(train_list)
    
    #train_work = train[names[-1]]
    test_list = []
    for line in open('../data/test_clean.json', 'r'):
        test_list.append(json.loads(line))
    test = pd.DataFrame(test_list)
    
    print('--- NLP on major, simply cut the first word')
    le = LabelEncoder()
    print len(set(train['major']))
    train['major'] = train['major'].apply(lambda x :  " ".join(jieba.cut(x, cut_all = False)).split()[0] if x is not None and len(" ".join(jieba.cut(x)).split()) > 0 else 'none')
    test['major']  = test['major'].apply(lambda x :  " ".join(jieba.cut(x,  cut_all = False)).split()[0] if x is not None  and len(" ".join(jieba.cut(x)).split()) > 0 else 'none')

    print len(set(train['major']))
    le.fit(list(train['major']) + list(test['major']))
    train['major'] = le.transform(train['major'])
    test['major'] = le.transform(test['major'])
 
    le = LabelEncoder()
    train['gender'] = le.fit_transform(train['gender'])
    names =  train.columns
    
    le = LabelEncoder()
    test['gender'] = le.fit_transform(test['gender'])
    del train['_id']
    del test['_id']
    train = train.fillna(0)
    test = test.fillna(0)
    #test['age'] = test['age'].apply(lambda x : int(x.replace(u'岁','').encode('ascii')))
    return train, test

def getPosDict():
    pos = dict()
    posdes = codecs.open('pos_dict.txt', 'rb',encoding = 'utf-8').readlines()
    for line in posdes:
        try:
            ch, en = line.strip().split(":")
            pos[ch] = en
        except ValueError:
            pos[ch] = 'manager'
        #print ch, pos[ch]
    return pos

def getMostSimilar(x):
    '''
        x is a chinese position_name
        new x is one of 32 position_name
    '''
    pos = getPosDict()
    try:
        en_x = pos[x]
        titles = en_x.decode('utf-8').encode('ascii', 'ignore')
        xvec = np.mean(np.array([ model[s.lower()] if s.lower() in model and s.lower() not in stopwords.words('english') else [0]* 100 for s in titles.strip().split() ]), 0)
        dots = [np.dot(xvec, jobvec[i, :]) for i in range(32)]
        #print dots, len(dots)
        index = np.argmax(np.array(dots))
        newx = jobs[index]
        print x, newx
    except KeyError:
        newx = u'销售经理'
    except UnicodeEncodeError:
        newx = u'销售经理'
    return newx

def preprocess_degree():
    if os.path.isfile('datasets_degree.pkl'):
        with open('datasets_degree.pkl', 'rb') as fp:
            newtrain = cPickle.load(fp)
            train_Y = cPickle.load(fp)
            le = cPickle.load(fp)
            newtest = cPickle.load(fp)
    else :

        print('--- reading input files')
        offset = 60000
        train, test = load_data()
        print('--- fill NaN')
        train = train.fillna(train.median())
        test = test.fillna(test.median())
        # train = DataFrameImputer().fit_transform(train)
        # #val = DataFrameImputer().fit_transform(val)
        # test = DataFrameImputer().fit_transform(test)
        names = train.columns
        print('--- Create Major Features')
        train = getFeature(train)
        test = getFeature(test)
        train, test = getFeatureTotal(train, test)
  
        # train['rank_pos'] = train['last_pos'].apply(lambda x : pos_dict[x])
        # test['rank_pos'] = test['last_pos'].apply(lambda x :   pos_dict[x])
        # enc = OneHotEncoder()
        # #enc.fit(np.array(train['rank_pos']))
        # train_rank_pos = enc.fit_transform(np.array(train['rank_pos'].reshape(70000,1)))
        # test_rank_pos  = enc.fit_transform(np.array(test['rank_pos'].reshape(20000,1)))
        #onehfeatures = ['rank_pos', 'last_salary', 'last_size', 'work_age', 'num_times_work', 'max_salary' ]

        #train_oneh , test_oneh = createOneHotFeature(train, test, onehfeatures)
   
        print train.columns

        train_degree = train['degree']
        train, val , train_y, val_y= train[:offset], train[offset:], train_degree[:offset], train_degree[offset:]
        train = train.drop(['id'], 1)
        val = val.drop(['id'], 1)
        test = test.drop(['id'], 1)
        # train = DataFrameImputer().fit_transform(train)
        # val = DataFrameImputer().fit_transform(val)
        # test = DataFrameImputer().fit_transform(test)
        print('generate log features of degree')
      
        train1, val1, test1 = getNewLogFeatures(train, val, test, "major", "degree")
        train2, val2, test2 = getNewLogFeatures(train, val, test, "last_salary", "degree")
        train3, val3, test3 = getNewLogFeatures(train, val, test, "last_pos", "degree")
        train4, val4, test4 = getNewLogFeatures(train, val, test, "age", "degree")
        train5, val5, test5 = getNewLogFeatures(train, val, test, "last_industry", "degree")
        train6, val6, test6 = getNewLogFeatures(train, val, test, "pre_pos", "degree")
        train7, val7, test7 = getNewLogFeatures(train, val, test, "pre_dep", "degree")

        num_features = len(train.columns)
        keep_list2 = train2.columns[num_features:]
        keep_list3 = train3.columns[num_features:]
        keep_list4 = train4.columns[num_features:]
        keep_list5 = train5.columns[num_features:]
        keep_list6 = train6.columns[num_features:]
        keep_list7 = train7.columns[num_features:]

        train  = pd.concat([train1, train2[keep_list2], train3[keep_list3],  train4[keep_list4],  train5[keep_list5], train6[keep_list6],train7[keep_list7] ], axis =1 )
        val   = pd.concat([val1, val2[keep_list2],  val3[keep_list3], val4[keep_list4], val5[keep_list5], val6[keep_list6], val7[keep_list7] ], axis =1 )
        test   = pd.concat([test1, test2[keep_list2], test3[keep_list3], test4[keep_list4], test5[keep_list5], test6[keep_list6], test7[keep_list7] ], axis =1 )

        train = train.drop('degree',1)
        val = val.drop('degree',1)    
        test = test.drop('degree',1)  

        train_tfidf, val_tfidf , test_tfidf = TFIDFeature(train, val, test, 'industry')  
        train = train.drop(['workExperienceList'], 1)
        val = val.drop(['workExperienceList'], 1)
        test = test.drop(['workExperienceList'], 1)
  
        create_feature_map(train)
        #train_oneh, val_oneh = train_oneh[:offset,:], train_oneh[offset:,:]
        # train = np.hstack([np.array(train), train_tfidf.toarray(), train_oneh])
        # val =   np.hstack([np.array(val)  , val_tfidf.toarray(),   val_oneh])
        # test=   np.hstack([np.array(test) , test_tfidf.toarray(),  test_oneh])

        train = np.array(train)
        val =  np.array(val)
        test  = np.array(test)
        xgtrain = xgb.DMatrix(train, label = train_y)
        xgval =  xgb.DMatrix(val, label = val_y)
        xgtest = xgb.DMatrix(test)
        watchlist = [(xgtrain, 'train'),(xgval, 'val')]
        params = {}
        params["objective"] = 'multi:softmax'
        params["eta"] = 0.1
        params["subsample"] = 0.7
        params["colsample_bytree"] = 0.8
        params["silent"] = 1
        params["max_depth"] = 8
        params["min_child_weight"] = 4
        params["gamma"] = 1
        params["num_class"] = 3
        params["eval_metric"] = 'merror'
        model = xgb.train(list(params.items()), xgtrain, 500, watchlist, early_stopping_rounds= 10)

        plot_importance(model, 'degree_feature_important_xgb.png')

        pred = model.predict(xgtest)
        sub = pd.read_csv('result/benchmark.csv')
        sub['degree'] = pd.Series([int(x) for x in pred])
        sub.to_csv('result/degree_pre.csv' ,encoding="utf-8", index=False)
        return pred

def preprocess_size():
    from sklearn.feature_selection import RFE
    if os.path.isfile('datasets_degree.pkl'):
        with open('datasets_degree.pkl', 'rb') as fp:
            newtrain = cPickle.load(fp)
            train_Y = cPickle.load(fp)
            le = cPickle.load(fp)
            newtest = cPickle.load(fp)
    else :
        print('--- reading input files')
        offset = 60000
        train, test = load_data()
    #     train = getFeature(train)
    #     test = getFeature(test)
    #     train ,test = getFeatureTotal(train, test)

    #     print('--- add size specfic features...')
    #     # train = getSizeFeatures(train)
    #     # test  = getSizeFeatures(test)

    #     train['size'] = train['workExperienceList'].apply(lambda x : x[1]['size']) - 1
    #     train,val,train_y, val_y = train[:offset], train[offset:], train['size'][:offset], train['size'][offset:]
    #     train = train.drop(['id'], 1)
    #     val = val.drop(['id'], 1)
    #     test = test.drop(['id'], 1)

    #     #featuresList = ['last_pos', 'last_job_long', 'last_size', 'last_salary', 'pre_pos', 'last_dep', 'pre_dep']
    #     featuresList = ['last_pos', 'pre_dep' ]
    #     train, val, test = getMultiLogFeatures(train, val, test, featuresList, "size")
        
    #     train = train.drop('size',1)
    #     val = val.drop('size',1) 
    #     #train_tfidf, val_tfidf , test_tfidf = TFIDFeature(train, val, test, 'industry')  
    #     train = train.drop(['workExperienceList'], 1)
    #     val = val.drop(['workExperienceList'], 1)
    #     test = test.drop(['workExperienceList'], 1)

    #     create_feature_map(train)
    #     # train = np.hstack([np.array(train), train_tfidf.toarray()])
    #     # val =   np.hstack([np.array(val)  , val_tfidf.toarray()])
    #     # test=   np.hstack([np.array(test) , test_tfidf.toarray()])
    #     train = np.array(train)
    #     val   = np.array(val)
    #     test  = np.array(test)

    #     xgtrain = xgb.DMatrix(train, label = train_y)
    #     xgval =  xgb.DMatrix(val, label = val_y)
    #     xgtest = xgb.DMatrix(test)
    #     watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    #     params = {}
    #     params["objective"] = 'multi:softmax'
    #     params["eta"] = 0.1
    #     params["subsample"] = 1
    #     params["colsample_bytree"] = 0.9
    #     params["silent"] = 1
    #     params["max_depth"] = 8
    #     params["min_child_weight"] = 4
    #     params["gamma"] = 1
    #     params["num_class"] = 7
    #     params["eval_metric"] = 'merror'
    #     model = xgb.train(list(params.items()), xgtrain, 1000, watchlist, early_stopping_rounds= 30)
    #     plot_importance(model, 'size_feature_important_xgb.png')
    #     pred = model.predict(xgtest) + 1
    #     # sle = StandardScaler()
    #     # sle.fit(train)
    #     # train  = sle.transform

        sub = pd.read_csv('result/merge.csv')
        #sub['size'] = pd.Series([int(x) for x in pred])
        sub['size'] = test['workExperienceList'].apply(lambda x : x[2]['size'])
        sub.to_csv('result/merge2.csv' , index=False)

def getNewLogFeatures(train, val, test,  nameA, nameC):
    logoddsCurrSalary, logoddsPCurrSalary, default_logodds_Sal = get_log_count(train, nameA, nameC)
    len_train = len(train.columns.values.tolist())
    train1 , val1 = generate_log_features(train, val, logoddsCurrSalary, logoddsPCurrSalary, nameA, nameC, default_logodds_Sal)
    train1, test1 = generate_log_features(train, test, logoddsCurrSalary, logoddsPCurrSalary, nameA, nameC, default_logodds_Sal)
    #return train1[len_train:], val1[len_train:], test1[len_train-1:]
    return train1, val1, test1

def preprocess_salary():

    print('--- reading input files')
    offset =25000
    train, test = load_data()
    print('--- fill NaN')
    train = train.fillna(-1)
    test = test.fillna(-1)
    # train = DataFrameImputer().fit_transform(train)
    # #val = DataFrameImputer().fit_transform(val)
    # test = DataFrameImputer().fit_transform(test)
    names = train.columns
    print('--- Create Major Features')
    train = getFeature(train)
    test = getFeature(test)

    train,test = getFeatureTotal(train, test)
    # train = getSizeFeatures(train)
    # test  = getSizeFeatures(test)
  
    train['salary'] = train['workExperienceList'].apply(lambda x : x[1]['salary'])        
 
    # train['position_name'] = train['workExperienceList'].apply(lambda x : x[1]['position_name'])
    # #---------------------- filtering out the non-32 positions
    # train = train.loc[train['position_name'].isin(jobs)]
    # train = train.drop(['position_name'],1)

    train, val , train_y, val_y = train[:offset], train[offset:], train['salary'][:offset], train['salary'][offset:]
    train = train.drop(['id'], 1)
    val = val.drop(['id'], 1)
    test = test.drop(['id'], 1)

    print('generate log features of position_name')
    #feature_list = ['last_salary', 'pre_salary']
    #train, val, test = getMultiLogFeatures(train, val, test, feature_list, 'salary')    
    
    train = train.drop('salary',1)
    val= val.drop('salary',1)  

    print train.columns

    #train_tfidf, val_tfidf, test_tfidf = TFIDFeature(train, val, test, 'last_pos')  
    train = train.drop(['workExperienceList'], 1)
    val = val.drop(['workExperienceList'], 1)
    test = test.drop(['workExperienceList'], 1)

    create_feature_map(train)
    train = np.array(train)
    val = np.array(val)
    test = np.array(test)

    xgtrain = xgb.DMatrix(train, label = train_y)
    xgval =  xgb.DMatrix(val, label = val_y)
    xgtest = xgb.DMatrix(test)
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]
    params = {}
    params["objective"] = 'multi:softmax'
    params["eta"] = 0.001
    params["subsample"] = 0.7
    params["colsample_bytree"] = 1
    params["silent"] = 1
    params["max_depth"] = 10
    params["min_child_weight"] = 100
    params["gamma"] = 2
    params["num_class"] = 7
    params["eval_metric"] = 'merror'
    model = xgb.train(list(params.items()), xgtrain, 3000, watchlist, early_stopping_rounds= 40)
    pred = model.predict(xgtest) 
    plot_importance(model, 'salary_feature_important_xgb.png')

    sub = pd.read_csv('result/benchmark.csv')
    sub['salary'] = pd.Series([int(x) for x in pred])
    sub.to_csv('result/salary_pred.csv' , index=False)


def preprocess_pos():
    if os.path.isfile('datasets_degree.pkl'):
        with open('datasets_degree.pkl', 'rb') as fp:
            newtrain = cPickle.load(fp)
            train_Y = cPickle.load(fp)
            le = cPickle.load(fp)
            newtest = cPickle.load(fp)
    else :
        print('--- reading input files')
        offset = 26000
        train, test = load_data()
        #---------------------- filtering out the non-32 positions
        train['position_name'] = train['workExperienceList'].apply(lambda x : x[1]['position_name'])
        train = train.loc[train['position_name'].isin(jobs)]
        
        names = train.columns
        print('--- Create Major Features')
        train = getFeature(train)
        test = getFeature(test)
        train, test = getFeatureTotal(train, test)
        train['salary'] = train['workExperienceList'].apply(lambda x : x[1]['salary'])        
 
        # train['rank_pos'] = train['last_pos'].apply(lambda x : pos_dict[x])
        # test['rank_pos'] = test['last_pos'].apply(lambda x :   pos_dict[x])
        # onehfeatures = ['work_age', 'last_salary', 'last_size' ]
        # train_oneh , test_oneh = createOneHotFeature(train, test, onehfeatures)
        
        le = LabelEncoder()
        train['position_name'] = le.fit_transform(train['position_name'])
        train, val , train_y, val_y = train[:offset], train[offset:], train['position_name'][:offset], train['position_name'][offset:]
        
        #train_tfidf, val_tfidf, test_tfidf = TFIDFeature(train, val, test, 'industry')  
        print('generate log features of position_name')
        feature_list = [ 'last_salary', 'last_pos', 'pre_pos' , 'last_size']
        train, val, test = getMultiLogFeatures(train, val, test, feature_list, 'position_name')
        # train, val, test = getMultiLogFeatures(train, val, test, feature_list, 'degree')
        # train, val, test = getMultiLogFeatures(train, val, test, feature_list, 'salary')
        
        train = train.drop(['id', 'salary'], 1)
        val = val.drop(['id', 'salary'], 1)
        test = test.drop(['id'], 1)

   
        #print('add TruncatedSVD and TSNE features..')
        train = train.drop('position_name',1)
        val = val.drop('position_name',1)    
        
        #train_tfidf, val_tfidf, test_tfidf = TFIDFeature(train, val, test, 'position_name')  
        # svd = TruncatedSVD(n_components=10, random_state=42)
        # train_svd =  svd.fit_transform(train_tfidf.toarray())
        # val_svd= svd.fit_transform(val_tfidf.toarray())
        # test_svd =svd.fit_transform(test_tfidf.toarray())
        # tsne = TSNE(n_components=3, random_state=0)
        # train_tsne = tsne.fit_transform(train_svd)
        # val_tsne =  tsne.fit_transform(val_svd)
        # test_tsne =  tsne.fit_transform(test_svd)
        # print train_tsne.shape

        train = train.drop(['workExperienceList'], 1)
        val = val.drop(['workExperienceList'], 1)
        test = test.drop(['workExperienceList'], 1)
        create_feature_map(train)
        
        #train_oneh, val_oneh = train_oneh[:offset,:], train_oneh[offset:,:]
        train = np.hstack([np.array(train)])
        val =   np.hstack([np.array(val)  ])
        test=   np.hstack([np.array(test) ])
        
        # train = np.hstack([np.array(train), train_tfidf.toarray()])
        # val = np.hstack([np.array(val), val_tfidf.toarray()])
        # test = np.hstack([np.array(test), test_tfidf.toarray()])
        print train.shape
        xgtrain = xgb.DMatrix(train, label = train_y)
        xgval =  xgb.DMatrix(val, label = val_y)
        xgtest = xgb.DMatrix(test)
        watchlist = [(xgtrain, 'train'),(xgval, 'val')]
        params = {}
        params["objective"] = 'multi:softmax'
        params["eta"] = 0.1
        params["subsample"] = 0.6
        params["colsample_bytree"] = 0.75
        params["silent"] = 1
        params["max_depth"] = 8
        params["min_child_weight"] = 5
        params["gamma"] = 1
        params["num_class"] = 32
        params["eval_metric"] = 'merror'

        model = xgb.train(list(params.items()), xgtrain, 800, watchlist, early_stopping_rounds= 30)
        pred = model.predict(xgtest) 
        #pred = np.argmax(pred, axis = 1)
        plot_importance(model, 'position_feature_important_xgb.png')
        pred = [int(x) for x in pred]
        sub = pd.read_csv('result/benchmark.csv')
        sub['position_name'] = le.inverse_transform(pred)#pd.Series([reverse_dict[x] for x in pred])
        sub.to_csv('result/submit_xgb.csv' , index=False)
        return pred

def plot_importance(model, fn):
    importance = model.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 6))
    # axes = plt.Axes(figure, [.2,.1,.7,.8]) # [left, bottom, width, height] where each value is between 0 and 1
    # figure.add_axes(axes) 
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')

    plt.gcf().savefig(fn)


def create_feature_map(train):
    features = list(train.columns[:30])
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def plot_ss_dis():
    train, test = load_data()
    train['size'] =train['workExperienceList'].apply(lambda x : x[1]['size'])
    train['salary'] = train['workExperienceList'].apply(lambda x : x[1]['salary'])
    train['size_salary'] = train['workExperienceList'].apply(lambda x : x[1]['size'] * 10 + x[1]['salary'])        
    plt.hist(train['size_salary'], bins=np.arange(0,79))
    #plt.scatter(train['salary'], train['size'])
    plt.show()
#train, test = load_data()
#benchmark(train)
#preprocess_degree()
#preprocess_size()
#preprocess_size_salary()
#pred = preprocess_binary_salary()
#preprocess_multi_salary(pred)
preprocess_salary()
#preprocess_pos()
#plot_ss_dis()