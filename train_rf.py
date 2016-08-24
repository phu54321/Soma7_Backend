#!/usr/bin/python
# -*- coding: utf-8 -*-

# Classify code from scikit-learn tutorial
# Text analysis: Working with text data

import pandas as pd
import numpy as np

from konlpy.tag import Mecab

from sklearn.externals import joblib
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

import sys
import codecs
import unicodedata

out = codecs.getwriter('utf-8')(sys.stdout)


def width(string):
    return sum(1 + (unicodedata.east_asian_width(c) in "WF")
               for c in string)

# Noun selector
mecab = Mecab()

print("Loading data")
df = pd.read_pickle('soma_goods_train.df')
print("Load completed")

data_count = len(df)
test_count = 0
train_count = data_count - test_count
# train_count = data_count
# test_count = data_count - train_count


def preprocess_df(df):
    df2 = pd.DataFrame(index=df.index, columns=['cate', 'origname', 'name'])
    df2['cate'] = df['cate1'] + ';' + df['cate2'] + ';' + df['cate3']

    namerow = []
    for index, row in df.iterrows():
        name = row['name'].lower()
        name = ' '.join(mecab.morphs(name))
        namerow.append(name)

    df2['name'] = pd.Series(namerow, index=df.index)
    df2['origname'] = df['name']

    return df2

df = preprocess_df(df)

print('Preprocessing data')
shuffled_df = df.sample(frac=1).reset_index(drop=True)
train_df = shuffled_df.head(train_count)
test_df = shuffled_df.tail(test_count)

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(
        loss='hinge',
        penalty='l2',
        n_iter=20)),
])

text_clf.fit(train_df['name'], train_df['cate'])

print('Testing accuracy')

if test_count > 0:
    docs_test = test_df['name']
    predicted = text_clf.predict(docs_test)
    print('Accuracy : %g' % np.mean(predicted == test_df['cate']))
    print(metrics.classification_report(
        test_df['cate'],
        predicted
    ))

    for name, pred, real in zip(test_df['origname'], predicted, test_df['cate']):
        print("%s %s%s  %s%s  %s" % (
            " " if pred == real else "X",
            real, ' ' * (45 - width(real)),
            pred, ' ' * (45 - width(pred)),
            name
        ))

joblib.dump(text_clf, 'text_clf.dat', compress=3)
