#
# Farfetech case study
#
# Product attribute classification with NLP
#
# Author: Kai Chen
# Date: Apr, 2018
#


import gc
import pandas as pd
import numpy as np
import os
import itertools
from random import shuffle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
# %matplotlib inline


from scipy.sparse import csr_matrix, hstack

from nltk.corpus import stopwords

from keras.preprocessing.text import text_to_word_sequence

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


np.random.seed(42)

# ---------------------
# Define the file paths
PRODUCT_CSV_FILE = 'data/products.csv'
ATTRIBUTE_CSV_FILE = 'data/attributes.csv'

# -----------
# Step 1. Read and explor the data
df_product = pd.read_csv(PRODUCT_CSV_FILE)

# print(df_product.head())
# print(df_product.shape)
# print(len(df_product['ProductId'].unique()))

list_product_id = df_product['ProductId'].unique()

# get product category
print('get product category.')
list_category = df_product['Category'].unique()
dict_product_cat = dict()
for product_id in list_product_id:
    category = df_product[df_product['ProductId'] == product_id]['Category'].values
    if (len(category)) > 1:
        print('>1')
    dict_product_cat[product_id] = category[0]

# get mapping product_id -> articlephoto_id
# ArticlePhotoId
list_photo_id = df_product['ArticlePhotoId'].unique()
print('photo id')
print(list_photo_id)
dict_photo_product_id = dict()
for photo_id in list_photo_id:
    dict_photo_product_id[photo_id] = df_product[df_product['ArticlePhotoId']==photo_id]['ProductId'].values[0]

df_attribute = pd.read_csv(ATTRIBUTE_CSV_FILE)
print(df_attribute.head())
print(df_attribute.shape)
print(len(df_attribute['ProductId'].unique()))

df_product_attribute = pd.merge(df_product, df_attribute, on=['ProductId'])
print(df_product_attribute.head())
print(df_product_attribute.shape)

print(len(df_product_attribute['ProductId'].unique()))
print(len(df_product_attribute['AttributeValueName'].unique()))

print(df_product_attribute.columns)
print(df_product_attribute.describe())

list_product_id = df_attribute['ProductId'].unique()
dict_product_des = dict()

print('get product description.')
for product_id in list_product_id:
    if product_id in dict_product_des:
        print('product {} has more than one description'.format(product_id))
    df_sub = df_product_attribute[df_product_attribute['ProductId'] == product_id]
    # print(df_sub['Description'].values[0])
    dict_product_des[product_id] = df_sub['Description'].values[0]

# create one-hot encoding for the attribute name
print('get product attribute.')
list_attribute = df_attribute['AttributeName'].unique()
dict_product_att = dict()
for product_id in list_product_id:
    dict_product_att[product_id] = dict()
    for attribute in list_attribute:
        dict_product_att[product_id][attribute] = 0
    list_product_attribute = df_product_attribute[df_product_attribute['ProductId'] == product_id]['AttributeName']
    # print(len(list_product_attribute))
    for attribute in list_product_attribute:
        dict_product_att[product_id][attribute] = 1

# ---------
# Step 2: Prepare
# prepare train, and test sets
shuffle(list_product_id)
percentage_train_set = 0.7
list_product_id_train = list_product_id[0:int(percentage_train_set*len(list_product_id))]
list_product_id_test = list_product_id[len(list_product_id_train):]

# print(len(list_product_id))
# print(len(list_product_id_train))
# print(len(list_product_id_test))

# -----------
# Step 4: NLP
# remove stop words

def cleanupDoc(s):
    stopset = set(stopwords.words('english'))
    stopset.add('wikipedia')
    tokens = text_to_word_sequence(s, filters="\"!'#$%&()*+,-˚˙./:;‘“<=·>?@[]^_`{|}~\t\n", lower=True, split=" ")
    cleanup = " ".join(filter(lambda word: word not in stopset, tokens))
    return cleanup


class_names = list_attribute

train_text = []
train_attribute = dict()

for class_name in class_names:
    train_attribute[class_name] = []

for product_id in list_product_id_train:
    train_text.append(dict_product_des[product_id])
    for class_name in class_names:
        train_attribute[class_name].append(dict_product_att[product_id][class_name])

test_text = []
test_attribute = dict()

for class_name in class_names:
    test_attribute[class_name] = []

for product_id in list_product_id_test:
    test_text.append(dict_product_des[product_id])
    for class_name in class_names:
        test_attribute[class_name].append(dict_product_att[product_id][class_name])

train_text = [cleanupDoc(text) for text in train_text]
test_text = [cleanupDoc(text) for text in test_text]


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 2),
    # max_features=50000,
    max_features=10000,
    )
train_word_features = word_vectorizer.fit_transform(train_text)
print('Word TFIDF 1/2')
test_word_features = word_vectorizer.transform(test_text)
print('Word TFIDF 2/2')

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    # max_features=50000,
    max_features=10000,
    )
train_char_features = char_vectorizer.fit_transform(train_text)
print('Char TFIDF 1/2')
test_char_features = char_vectorizer.transform(test_text)
print('Char TFIDF 2/2')

train_features = hstack([train_char_features, train_word_features])
print('HStack 1/2')
test_features = hstack([test_char_features, test_word_features])
print('HStack 2/2')

pred_attribute = dict()
scores = []
for class_name in class_names:
    # print(class_name)
    train_target = train_attribute[class_name]

    classifier = LogisticRegression(solver='sag')
    sfm = SelectFromModel(classifier, threshold=0.2)

    # print(train_features.shape)
    train_sparse_matrix = sfm.fit_transform(train_features, train_target)
    # print(train_sparse_matrix.shape)

    # train_sparse_matrix, valid_sparse_matrix, y_train, y_valid = train_test_split(train_sparse_matrix, train_target,
    #                                                                               test_size=0.05, random_state=42)
    test_sparse_matrix = sfm.transform(test_features)

    cv_score = np.mean(cross_val_score(classifier, train_sparse_matrix, train_target, cv=2, scoring='roc_auc'))
    scores.append(cv_score)
    # print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_sparse_matrix, train_target)

    pred_attribute[class_name] = classifier.predict_proba(test_sparse_matrix)[:, 1]



# ---------
# Future work
