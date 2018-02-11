#encoding=UTF-8
#读取数据   单词向量化   词频-逆文档频率
import pyprind
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
from nltk.stem.porter import PorterStemmer


'''
np.random.seed(0)
pbar=pyprind.ProgBar(50000);
labels={'pos':1,'neg':0}
df=pd.DataFrame()

for s in ('test','train'):
    for l in ('pos','neg'):
        path='./aclImdb/%s/%s' % (s,l)
        for file in os.listdir(path):
            with open(os.path.join(path,file),'r') as infile:
                txt=infile.read()
            df=df.append([[txt,labels[l]]],ignore_index=True)
            pbar.update()

df.columns=['review','sentiment']

df=df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv',index=False)

print df.head(3)
'''

df=pd.read_csv('./movie_data.csv')
count=CountVectorizer()

docs=np.array([
                'The sun is shining',
                'The weather is sweet',
                'The sun is shining and the weather is sweet'
              ])
bag=count.fit_transform(docs)
tfidf=TfidfTransformer()
np.set_printoptions(precision=2)

print (count.vocabulary_)

print bag.toarray()

print (tfidf.fit_transform(count.fit_transform(docs)).toarray())