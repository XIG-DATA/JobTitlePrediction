#encoding=utf8
import pandas as pd
import numpy as np
import json
from copy import deepcopy
from sklearn.base import BaseEstimator
import sys
from sklearn_pandas import DataFrameMapper
reload(sys)
sys.setdefaultencoding('utf-8') 

class GoldenFeature(BaseEstimator):
    def __init__(self,data,yname="predict_degree",use_minmax=False):
        self.use_minmax = use_minmax
        self.data = data
        self.yname = yname
        self.pred_labels = np.unique(data[yname])
        self.minmax=None

    def fit(self,df):
        data =self.data
        yname = self.yname
        cname = pd.DataFrame(df).columns[0]
        self.cname = cname
        self.feature_dict = dict()
        column_labels = np.unique(df)
        d = pd.merge(pd.DataFrame(df),pd.DataFrame(data[yname]),left_index=True,right_index=True,how='inner')

        columns = sorted(d[cname].unique())
        targets = sorted(d[yname].unique())

        t_counts=data.groupby([yname]).size()
        c_t_counts=data.groupby([cname,yname]).size()
        c_counts=data.groupby([cname]).size()
        logodds={}
        logoddsPA={}
        MIN_CAT_COUNTS=2
        default_logodds=np.log(t_counts/float(len(data)))-np.log(1.0-t_counts/float(len(data)))
        logodds['default']=deepcopy(default_logodds)
        logoddsPA['default']=-99
        for col in columns:
            PA=c_counts[col]/float(len(data))
            logoddsPA[col]=np.log(PA)-np.log(1.-PA)
            logodds[col]=deepcopy(default_logodds)
            for cat in c_t_counts[col].keys():
                if (c_t_counts[col][cat]>MIN_CAT_COUNTS) and c_t_counts[col][cat]<c_counts[col]:
                    PA=c_t_counts[col][cat]/float(c_counts[col])
                    logodds[col][targets.index(cat)]=np.log(PA)-np.log(1.0-PA)
            logodds[col]=pd.Series(logodds[col])
            logodds[col].index=range(len(targets))

        self.logodds=logodds
        self.logoddsPA = logoddsPA


    def transform(self,df):
        logodds= self.logodds
        logoddsPA = self.logoddsPA
        cname = pd.DataFrame(df).columns[0]
        new_features = df.apply(lambda x: logodds[x] if x in logodds else logodds['default'])
        new_features.columns=["logodds"+str(x) for x in range(len(new_features.columns))]
        new_PA_features = df.apply(lambda x: logoddsPA[x] if x in logoddsPA else logoddsPA['default'])
        new_features=pd.merge(pd.DataFrame(new_features),pd.DataFrame(new_PA_features),left_index=True,right_index=True,how='inner')
        return new_features

train_list = []
for line in open('../data/train_clean.json', 'r'):
    train_list.append(json.loads(line))
train = pd.DataFrame(train_list)
train = train.fillna(-1)
gf = GoldenFeature(train,"degree",False)
print train
gf.fit(train['age'])
newv = gf.transform(train['age'])

print newv

