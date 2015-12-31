from sklearn.base import BaseEstimator, TransformerMixin
#from nltk import word_tokenize
#from nltk.stem import WordNetLemmatizer
from types import NoneType
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import jieba, itertools, json, codecs
jobs = codecs.open('../data/pos_name.txt', 'rb', encoding = 'utf-8').readlines()
jobs = [ job.strip() for job in jobs]

class ExtractPos():
    """ 
    Class that extracts recipe information from JSON.
    """
    def __init__(self, txt):
        self.id = self.set_id(json.loads(txt))
        self.work = 'workExperienceList'
        self.pos = self.set_pos(json.loads(txt))
        self.other_pos = self.set_other_pos(json.loads(txt))
        self.gender = self.set_gender(json.loads(txt))
        self.age = self.set_age(json.loads(txt))
        self.pos_count = len(self.other_pos)
        self.major = self.set_major(json.loads(txt))
        # set last-series parameters        
        self.last_pos = self.set_last_pos(json.loads(txt))
        self.last_industry = self.set_last_industry(json.loads(txt))
        self.last_department = self.set_last_department(json.loads(txt))
        self.last_salary = self.set_last_salary(json.loads(txt))
        self.last_size = self.set_last_size(json.loads(txt))
        # set previous-series parameters

        self.pre_pos = self.set_pre_pos(json.loads(txt))
        self.pre_industry = self.set_pre_industry(json.loads(txt))
        self.pre_department = self.set_pre_department(json.loads(txt))
        self.pre_salary = self.set_pre_salary( json.loads(txt))
        self.pre_size =  self.set_pre_size(json.loads(txt))
        # set golden features

        # set other relevant features
        self.work_age = self.set_work_age(json.loads(txt))
        self.last_job_long= self.set_last_job_long(json.loads(txt))
        self.pre_job_long = self.set_pre_job_long(json.loads(txt))
        self.num_times_work = self.set_num_times_work(json.loads(txt))
        self.num_unique_work = self.set_num_unique_work( json.loads(txt))
        
    def __str__(self):
        return "ID: %s\nPosition: %s\nIndustry: %s\nNumber of Pos: %s" % (self.id, self.pos,', '.join(self.industry),self.pos_count)
    
    def set_id(self,txt):
        """
        Method that sets the recipe id.
        """
        try:
            return txt['id']
        except KeyError:
            return '-99'

    def get_interval(self, x, i, j):
        r = 0
        try:
            end_year =  int(x[i]['end_date'][:4])
        except:
            end_year = 2015
        try:
            end_month =  int(x[i]['end_date'][5:])
            start_year =  int(x[j]['start_date'][:4])
            start_month  = int(x[j]['start_date'][5:])
            r = (end_year - start_year) * 12 + end_month - start_month
        except:
            r = -1
        return r

    def work_total_month(self, x):
        try:
            end_year =  int(x[0]['end_date'][:4])
        except:
            end_year = 2015
        try:
            end_month =  int(x[0]['end_date'][5:])
            start_year =  int(x[-1]['start_date'][:4])
            start_month  = int(x[-1]['start_date'][5:])
            r = (end_year - start_year) * 12 + end_month - start_month
        except:
            r = -1
        return r

    def num_unique_work(self,x):
        x = [ x[i] for i in range(len(x)) if i!=1 ]
        xlist = [ xx['position_name']  if xx is not None else 'none' for xx in x  ]
        return len(set(xlist))

    def set_major(self,txt):
        return txt['major']

    def set_pos(self,txt):
        """
        Method that sets the second last position.
        """
        x = txt[self.work][1]
        if x is None:
            return ''
        else:
            return x['position_name']

    def set_other_pos(self, txt):
        """
        Method that sets other positions.
        """
        try:
            list_pos = [jieba.cut(txt[self.work][i]['position_name'], cut_all = False) if i !=1 and txt[self.work][i]['position_name'] is not None else 'none' for i in range(len(txt[self.work])) ]
            list_industry = [jieba.cut(txt[self.work][i]['industry'], cut_all = False) if i !=1 and txt[self.work][i]['industry'] is not None else 'none' for i in range(len(txt[self.work])) ]
            list_depart = [ jieba.cut(txt[self.work][i]['department'], cut_all = False) for i in range(len(txt[self.work])) if i !=1 and not isinstance(txt[self.work][i]['department'], NoneType) ]
            #list_major = [ jieba.cut(txt['major'], cut_all= False)] if txt['major'] is not None  else ['none']
            return list(itertools.chain(*list_pos)) + list(itertools.chain(*list_industry)) + list(itertools.chain(*list_depart))
        except TypeError:
            return 'none'
    def set_gender(self,txt):
        """
        Method that sets the gender
        """
        try:
            return txt['gender']
        except KeyError:
            return 'none'

    def set_age(self,txt):
        """
        Method that sets the age
        """
        try:
            return 0 if len(txt['age'].encode('ascii','ignore')) == 0 else int(txt['age'].encode('ascii', 'ignore'))
        except KeyError:
            return 0

    def set_work_age(self, txt):
        """
        """
        try:
            return  self.work_total_month(txt[self.work])
        except KeyError:
            return 0

    def set_last_job_long(self, txt):
        try:
            return  self.get_interval( txt[self.work], 0, 0)
        except KeyError:
            return 0

    def set_pre_job_long(self, txt):
        try: 
            return self.get_interval( txt[self.work], 2, 2)
        except KeyError:
            return 0

    def set_num_times_work(self, txt):
        try:
            return  len(txt[self.work])-1
        except KeyError:
            return 0

    def set_num_unique_work(self, txt):
        try:
            return self.num_unique_work(txt[self.work])
        except KeyError:
            return 0

    def set_last_pos(self, txt):
        """
        Method that sets the last posiiton_name
        """
        return txt[self.work][0]['position_name']
    
    def set_pre_pos(self, txt):
        """
        Method that sets the previous posiiton_name
        """
        return txt[self.work][2]['position_name'] if txt[self.work][2]['position_name'] is not None else 'none'
       
    def set_last_salary(self,txt):
        """
        Method that sets the last salary
        """
        return txt[self.work][0]['salary']
       
    def set_pre_salary(self,txt):
        """
        Method that sets the previous salary
        """
        return txt[self.work][2]['salary'] if txt[self.work][2]['salary'] is not None else 0

    def set_last_size(self,txt):
        """
        Method that sets the last size
        """
        return txt[self.work][0]['size']

    def set_pre_size(self,txt):
        """
        Method that sets the previous size
        """
        return txt[self.work][2]['size'] if txt[self.work][2]['size']  is not None else 0

    def set_last_industry(self,txt):
        """
        Method that sets the last industry
        """
        x = txt[self.work][0]['industry']
        if x is None:
            return 'none'
        return x

    def set_pre_industry(self,txt):
        """
        Method that sets the previous industry
        """
        return  txt[self.work][2]['industry'] if  txt[self.work][2]['industry'] is not None else 'none'

    def set_last_department(self,txt):
        """
        Method that sets the last department
        """
        return txt[self.work][0]['department'] if txt[self.work][0]['department'] is not None else 'none'
        
    def set_pre_department(self,txt):
        """
        Method that sets the previous department
        """
        return txt[self.work][2]['department'] if txt[self.work][2]['department'] is not None else 'none'

    def clean_ingredient(self,s):
        """
        Method that returns a cleaned up version of the entered ingredient.
        """
        from re import sub
        return sub('[^A-Za-z0-9]+', ' ', s)
    def get_train(self):
        """
        Method that returns a dictionary of data for the training set.
        """
        return {
            'pos':self.pos,
            'gender': self.gender,
            'age': self.age,
            'major':self.major,
            'last_pos' : self.last_pos,
            'last_salary' : self.last_salary,
            'last_size' : self.last_size,
            'last_industry' : self.last_industry,
            'last_department' : self.last_department,
            'pre_pos' : self.pre_pos,
            'pre_salary' : self.pre_salary,
            'pre_size' : self.pre_size,
            'pre_industry' : self.pre_industry,
            'pre_department' : self.pre_department,
            'other_pos': ', '.join(self.other_pos),
            'pos_count':self.pos_count,
            'work_age' : self.work_age,
            'last_job_long' : self.last_job_long,
            'pre_job_long' :  self.pre_job_long,
            'num_unique_work' : self.num_unique_work,
            'num_times_work' : self.num_times_work

        }
    def get_predict(self):
        """
        Method that returns a dictionary of data for predicting pos.
        """
        return {
            'id':self.id,
            'gender': self.gender,
            'age': self.age,
            'major': self.major,
            'last_pos' : self.last_pos,
            'last_salary' : self.last_salary,
            'last_size' : self.last_size,
            'last_industry' : self.last_industry,
            'last_department' : self.last_department,
            'pre_pos' : self.pre_pos,
            'pre_salary' : self.pre_salary,
            'pre_size' : self.pre_size,
            'pre_industry' : self.pre_industry,
            'pre_department' : self.pre_department,
            'other_pos':', '.join([x for x in self.other_pos]),
            'pos_count':self.pos_count,
            'work_age' : self.work_age,
            'last_job_long' : self.last_job_long,
            'pre_job_long' :  self.pre_job_long,
            'num_unique_work' : self.num_unique_work,
            'num_times_work' : self.num_times_work
        }   

class IngredientModel():
    """
    Class that stores an ingredient to pos model.
    """
    def __init__(self,model):
        self.model = model
    def predict(self,X):
        from pandas import Series
        from operator import add
        return X.other_pos.str.split(',? ').apply(lambda recipe: Series(reduce(add,[self.model.predict_proba([x]) for x in recipe])[0]/len(recipe)))

class TextModel():
    """
    Class that stores and a simple weighted average of two text-based individual  models.
    """
    def __init__(self,a_model,b_model):
        self.a_model = a_model
        self.b_model = b_model
        self.a_weight = 0.5
        self.b_weight = 0.5
    def set_weights(self,a_weight,b_weight):
        self.a_weight = a_weight
        self.b_weight = b_weight
    def blend(self,a_pred,b_pred):
        return a_pred*self.a_weight + b_pred*self.b_weight
    def predict(self,X):
        a_pred = self.a_model.predict_proba(X)[:,1]
        b_pred = self.b_model.predict_proba(X)[:,1]
        return self.blend(a_pred,b_pred)

class RecipeModel():
    """
    Class that stores the models needed to predict the type of pos based on a list of other_pos.
    """
    def __init__(self,ingred_model,text_models,recipe_model_a,recipe_model_b,encoder):
        self.ingred_model = ingred_model
        self.text_models = text_models
        self.recipe_model_a = recipe_model_a
        self.recipe_model_b = recipe_model_b
        self.recipe_weight_a = 0.5
        self.recipe_weight_b = 0.5
        self.score = 0.0
        self.encoder = encoder
    def __str__(self):
        return "\nRecipe Model\nBlended Accuracy: %0.5f\nModel A Weight: %0.2f\nModel B Weight: %0.2f" % (self.score, self.recipe_weight_a, self.recipe_weight_b)
    def set_weights(self,pred_a,pred_b,target):
        from sklearn.metrics import accuracy_score
        for w in zip(range(1,100,1),range(99,0,-1)):
                score = accuracy_score(target,(pred_a*w[0]/100.0+pred_b*w[1]/100.0).argmax(1))
                if score > self.score:
                    self.recipe_weight_a = w[0]/100.0
                    self.recipe_weight_b = w[1]/100.0
                    self.score = score
    def predict_kaggle(self,X,prob=False):
        # add average ingredient scores for each pos
        X = X.join(self.ingred_model.predict(X))
        # add pos based text models
        for v in self.text_models.keys():
            X['pred_text_'+v] = self.text_models[v].predict(X.other_pos)
        # make prediction for recipe model
        pred_a = self.recipe_model_a.predict_proba(X)
        pred_b = self.recipe_model_b.predict_proba(X)
        pred = pred_a*self.recipe_weight_a + pred_b*self.recipe_weight_b
        if prob:
            return pred
        else:
            return self.encoder.inverse_transform(pred.argmax(1))
    def predict(self,json_list,prob=False):
        """
        Return: The predicted pos for the list of recipes. 
        Params:
            * json_list (List of Dicts): The list of JSON recipes seeking pos predictions. 
            * prob: (Boolean) If the output should be the predicted probability across all poss or the best guess label. Defaults to False. 
        Doctest:
        >>> json_list = [
        ...     {
        ...             'id':1,
        ...             'other_pos': ['pork, black beans, avocado, orange, cumin, salt, cinnamon']
        ...     },
        ...     {
        ...             'id':2,
        ...             'other_pos': ['pasta, basil, pine nuts, olive oil, parmesan cheese, garlic']
        ...     },
        ...     {
        ...             'id':3,
        ...             'other_pos': ['tumeric, red lentils, naan, garam masala, onions, sweet potatoes']
        ...     }
        ... ]
        >>> recipe_model.predict(json_list)
        array([u'mexican', u'italian', u'indian'], dtype=object)
        >>> recipe_model.predict(json_list,prob=True)
        array([[  3.30066887e-03,   1.99567227e-06,   1.69680381e-03,
                  1.05174537e-05,   2.03085874e-05,   3.29112324e-03,
                  1.15989352e-06,   4.94386325e-03,   3.40759867e-06,
                  5.00931910e-03,   1.64480210e-03,   1.65024605e-03,
                  7.54782889e-07,   9.33262747e-01,   5.83895764e-07,
                  3.17332176e-06,   2.34691288e-02,   1.01873500e-02,
                  6.56450009e-03,   4.93754585e-03],
               [  1.48577580e-05,   4.19468051e-06,   1.68586413e-03,
                  1.27146558e-05,   8.20049665e-06,   1.16352625e-02,
                  3.69138042e-02,   3.23459014e-05,   1.23802178e-04,
                  5.19810867e-01,   1.16872025e-05,   1.17136206e-04,
                  6.88425371e-06,   4.24320803e-01,   1.23880210e-05,
                  1.07642463e-05,   3.49156883e-03,   1.75572406e-03,
                  2.47755352e-05,   6.35768902e-06],
               [  3.28104312e-03,   1.04083038e-05,   1.76803405e-06,
                  1.64068986e-03,   9.84115643e-03,   1.66666236e-03,
                  1.31338270e-02,   6.97641581e-01,   3.29681417e-03,
                  1.31455590e-02,   2.93287824e-06,   7.81646840e-02,
                  3.98645284e-07,   4.43144651e-02,   1.05028641e-01,
                  2.53745611e-06,   2.05663297e-02,   1.67476028e-03,
                  6.58238004e-03,   3.37299002e-06]])
        """
        from pandas import DataFrame
        # extract features from JSON
        X = DataFrame([ExtractPos(x).get_predict() for x in json_list])
        # add average ingredient scores for each pos
        X = X.join(self.ingred_model.predict(X))
        # add pos based text models
        for v in self.text_models.keys():
            X['pred_text_'+v] = self.text_models[v].predict(X.other_pos)
        # make prediction for recipe model
        pred_a = self.recipe_model_a.predict_proba(X)
        pred_b = self.recipe_model_b.predict_proba(X)
        pred = pred_a*self.recipe_weight_a + pred_b*self.recipe_weight_b
        if prob:
            return pred
        else:
            return self.encoder.inverse_transform(pred.argmax(1))
        
class VarSelect(BaseEstimator, TransformerMixin):
    def __init__(self, keys):
        self.keys = keys
    def fit(self, x, y=None):
        return self
    def transform(self, df):
        return df[self.keys]

# class LemmaTokenizer(object):
#   def __init__(self):
#       self.wnl = WordNetLemmatizer()
#   def __call__(self, doc):
#       return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def stripString(s):
    return ', '.join([''.join(y for y in x if y.isalnum()) for x in s.split(',')])

def loadTrainSet(dir='../data/train_clean.json'):
    """
    Read in txt to create training set.
    """
    import codecs
    from pandas import DataFrame, Series
    from sklearn.preprocessing import LabelEncoder
    #data = open(dir,'rb').readlines()
    X = DataFrame([ExtractPos(x).get_train() for x in open(dir,'rb') if (json.loads(x))['workExperienceList'][1]['position_name'] in jobs  ])   
    encoder = LabelEncoder()
    X['gender'] = encoder.fit_transform(X['gender'])
    # X['last_pos'] = encoder.fit_transform(X['last_pos'])
    # X['last_industry'] = encoder.fit_transform(X['last_industry'])
    # X['last_department'] = encoder.fit_transform(X['last_department'])
    # X['pre_pos'] = encoder.fit_transform(X['pre_pos'])
    # X['pre_industry'] = encoder.fit_transform(X['pre_industry'])
    # X['pre_department'] = encoder.fit_transform(X['pre_department'])

    encoder = LabelEncoder()
    X['pos'] = encoder.fit_transform(X['pos'])
    
    return X, encoder

def loadTestSet(dir='../data/test_clean.json'):
    """
    Read text to create test set.
    """
    import codecs
    from pandas import DataFrame
    X = DataFrame([ExtractPos(x).get_predict() for x in open(dir,'rb')])
    encoder = LabelEncoder()
    X['gender'] = encoder.fit_transform(X['gender'])
    # X['last_pos'] = encoder.fit_transform(X['last_pos'])
    # X['last_industry'] = encoder.fit_transform(X['last_industry'])
    # X['last_department'] = encoder.fit_transform(X['last_department'])
    # X['pre_pos'] = encoder.fit_transform(X['pre_pos'])
    # X['pre_industry'] = encoder.fit_transform(X['pre_industry'])
    # X['pre_department'] = encoder.fit_transform(X['pre_department'])

    return X

def transformLabel(train,test, nameA):
    le = LabelEncoder()
    le.fit(list(train[nameA]) + list(test[nameA]))
    train[nameA] = le.transform(train[nameA])
    test[nameA] = le.transform(test[nameA])
    return train, test
    
def loadData():
    train, encoder = loadTrainSet()
    test = loadTestSet()

    xlist = ['major', 'last_pos', 'last_industry', 'last_department', 'pre_pos', 'pre_department', 'pre_industry']
    for x in xlist:
        train,test = transformLabel(train,test, x)
    return train, encoder, test

def fitSklearn(X,y,cv,i,model, flag , multi=False):
    """
    Train a sklearn pipeline or model -- wrapper to enable parallel CV.
    """
    tr = cv[i][0]
    vl = cv[i][1]
    if flag == 1:
        model.fit(X.iloc[tr],y.iloc[tr])
    else:
        model.fit(X.iloc[tr], y.iloc[tr])
    if multi:
        return  {"pred": model.predict_proba(X.iloc[vl]), "index":vl}
    else:
        return  {"pred": model.predict_proba(X.iloc[vl])[:,1], "index":vl}

def trainSklearn(model,grid,train,target,cv,refit=True,n_jobs=1,multi=False):
    """
    Train a sklearn pipeline or model using textual data as input.
    """
    from joblib import Parallel, delayed   
    from sklearn.grid_search import ParameterGrid
    from numpy import zeros
    if multi:
        pred = zeros((train.shape[0],target.unique().shape[0]))
        from sklearn.metrics import accuracy_score
        score_func = accuracy_score
    else:
        from sklearn.metrics import roc_auc_score
        score_func = roc_auc_score
        pred = zeros(train.shape[0])
    best_score = 0
    for g in ParameterGrid(grid):
        model.set_params(**g)
        if len([True for x in g.keys() if x.find('nthread') != -1 ]) > 0:
            results = [fitSklearn(train,target,list(cv),i,model, 0, multi) for i in range(cv.n_folds)]
        else:
            results = Parallel(n_jobs=n_jobs)(delayed(fitSklearn)(train,target,list(cv),i,model, 0, multi) for i in range(cv.n_folds))
        if multi:
            for i in results:
                pred[i['index'],:] = i['pred']
            score = score_func(target,pred.argmax(1))
        else:
            for i in results:
                pred[i['index']] = i['pred']
            score = score_func(target,pred)
        if score > best_score:
            best_score = score
            best_pred = pred.copy()
            best_grid = g
    print "Best Score: %0.5f" % best_score 
    print "Best Grid", best_grid
    if refit:
        model.set_params(**best_grid)
        model.fit(train,target)
    return best_pred, model

def trainText(model_a,modelGrid_a,model_b,modelGrid_b,train,target,cv,refit=True,n_jobs=1):
    """
    Train and blend two univariate text models.
    """
    from sklearn.metrics import roc_auc_score
    from copy import deepcopy
    pred_a, model_a = trainSklearn(deepcopy(model_a),modelGrid_a,train,target,cv,refit=refit,n_jobs=n_jobs)
    pred_b, model_b = trainSklearn(deepcopy(model_b),modelGrid_b,train,target,cv,refit=refit,n_jobs=n_jobs)
    models = TextModel(model_a,model_b)
    best_score = 0
    for w in zip(range(2,100,2),range(98,0,-2)):
        score = roc_auc_score(target,pred_a*w[0]/100.0+pred_b*w[1]/100.0)
        if score > best_score:
            best_score = score
            models.set_weights(w[0]/100.0,w[1]/100.0)
    final_pred = models.blend(pred_a,pred_b)
    print "A Weight:",models.a_weight
    print "B Weight:", models.b_weight
    print "Best Blended Score: %0.5f" % roc_auc_score(target,final_pred)
    return final_pred, models

def splitother_pos(X):
    from pandas import Series
    X2 = X.other_pos.str.split(',? ').apply(lambda x: Series(x)).stack().reset_index(level=1, drop=True)
    X2.name = 'other_position'
    return X[['pos']].join(X2)

def fitIngredients(X,cv,i,model):
    """
    Train a sklearn pipeline or model -- wrapper to enable parallel CV.
    """
    from operator import add
    from pandas import Series
    tr = cv[i][0]
    vl = cv[i][1]
    X2 = splitother_pos(X.iloc[tr])
    model.fit(X2.other_position,X2.pos)
    return  {"pred":X.iloc[vl].other_pos.str.split(',? ').apply(lambda recipe:  Series(reduce(add,[model.predict_proba([x]) for x in recipe])[0]/len(recipe))), "index":vl}

def trainIngredient(model,grid,train,cv,refit=True,n_jobs=1):
    from joblib import Parallel, delayed   
    from sklearn.grid_search import ParameterGrid
    from numpy import zeros
    from sklearn.metrics import accuracy_score
    pred = zeros((train.shape[0],train.pos.unique().shape[0]))
    best_score = 0
    for g in ParameterGrid(grid):
        model.set_params(**g)
        results = Parallel(n_jobs=n_jobs)(delayed(fitIngredients)(train,list(cv),i,model) for i in range(cv.n_folds))
        for i in results:
            pred[i['index'],:] = i['pred']
        score = accuracy_score(train.pos,pred.argmax(1))
        if score > best_score:
            best_score = score
            best_pred = pred.copy()
            best_grid = g
    print "Best Score: %0.5f" % best_score 
    print "Best Grid", best_grid
    if refit:
        X2 = splitother_pos(train)
        model.set_params(**best_grid)
        model.fit(X2.other_position,X2.pos)
    return best_pred, IngredientModel(model)
    
def trainFeatureModel(train,target,model,grid,cv, flag , n_jobs=1):
    from sklearn.grid_search import ParameterGrid
    from sklearn.metrics import accuracy_score
    from joblib import Parallel, delayed  
    from numpy import zeros
    pred = zeros((train.shape[0],target.unique().shape[0]))
    best_score = 0
    best_grid = {}
    for g in ParameterGrid(grid):
        model.set_params(**g)
        if len([True for x in g.keys() if x.find('nthread') != -1 or x.find('n_jobs') != -1 ]) > 0:
            results = [fitSklearn(train,target,list(cv),i,model, flag, True) for i in range(cv.n_folds)]
        else:
            results = Parallel(n_jobs=n_jobs)(delayed(fitSklearn)(train,target,list(cv),i,model,flag,True) for i in range(cv.n_folds))
        for i in results:
            pred[i['index'],:] = i['pred']
        score = accuracy_score(target,pred.argmax(1))
        if score > best_score:
            best_score = score
            best_pred = pred.copy()
            best_grid = g
    print "Best Score: %0.5f" % best_score 
    print "Best Grid:", best_grid
    model.set_params(**best_grid)
    model.fit(train,target)
    return best_pred, model

    

