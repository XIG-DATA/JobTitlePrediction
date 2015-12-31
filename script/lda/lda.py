import jieba, os
from gensim import corpora, models, similarities
train_set = []


# walk = os.walk('./baike_topic')
# for root, dirs, files in walk:
#     for name in files:
#         f = open(os.path.join(root, name), 'r')
#    raw = f.read()
#    word_list = list(jieba.cut(raw, cut_all = False))
#    train_set.append(word_list)
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

def data_transform():
	train, test = load_data()
	train_set = []
	train['major_cut'] = train['major'].apply(lambda x : list(jieba.cut(x, cut_all = False)))
	return train['major_cut'].values.tolist()

train_set = data_transform()
dic = corpora.Dictionary(train_set)
corpus = [dic.doc2bow(text) for text in train_set]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lda = models.LdaModel(corpus_tfidf, id2word = dic,iterations=100, num_topics = 12)
corpus_lda = lda[corpus_tfidf]