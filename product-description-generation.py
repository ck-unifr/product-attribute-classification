# Farfetech case study
#
# Product description generation
# The object of this script is using deep learning technologies (CNN) for product category classification
#
# Author: Kai Chen
# Date: Apr, 2018
#

from pickle import dump
from pickle import load

import string

import pandas as pd
import numpy as np
import os
import sys
import itertools
import operator
from random import shuffle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline

from pickle import load

from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras import callbacks, applications, optimizers
from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

from nltk.translate.bleu_score import corpus_bleu

np.random.seed(42)

# Reference
# https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/

# ---------------------
# Define the file paths
PRODUCT_CSV_FILE = 'data/products.csv'
ATTRIBUTE_CSV_FILE = 'data/attributes.csv'

plot_figure = False


# -----------
# Step 1. Read and explore the data

df_product = pd.read_csv(PRODUCT_CSV_FILE)

# print(df_product.head())
# print(df_product.shape)
# print(len(df_product['ProductId'].unique()))

list_product_id_df = df_product['ProductId'].unique()
list_product_id_df = np.array(list_product_id_df)
print('number of products {} in the csv file'.format(list_product_id_df.shape[0]))

# create a dictionary with key: photo id -> value: product id
# note one photo belongs only to one product
list_photo_id = df_product['ArticlePhotoId'].unique()
dict_photo_product_id = dict()
for photo_id in list_photo_id:
    dict_photo_product_id[photo_id] = df_product[df_product['ArticlePhotoId']==photo_id]['ProductId'].values[0]

# update the list_product_id, such that each product should have an image
list_product_id = []
# img_width, img_height = 100, 100
# img_dir_path = "data/images_{}_{}/".format(img_width, img_height)
# img_width, img_height = 100, 100
img_dir_path = "data/images/"

dirs = os.listdir(img_dir_path)

for file_name in dirs:
    file_path = os.path.join(img_dir_path, file_name)
    product_id = int(file_name.split('_')[0])

    if not product_id in list_product_id_df:
        print('photo {} does not have product information'.format(file_path))
    else:
        list_product_id.append(product_id)

# print('PRODUCT IDS')
# for product_id in list_product_id:
#     print(product_id)


prepare_img_data = True
prepare_text_data = True

TRAIN_MODEL = True
EVALUATE_MODEL = True

# ---------
# Step 2: Extract image and text information for each product

# img_width, img_height = 80, 80
# img_width, img_height = 100, 100
img_width, img_height = 224, 224

# img_dir_path = "data/images_{}_{}/".format(img_width, img_height)
img_dir_path = "data/images/"
dirs = os.listdir(img_dir_path)

# ---------
# Prepare Image Data
product_image_feature_file_path = 'product-vgg-features.pkl'

# extract VGG16 features
def extract_features(dict_product_img):
    # model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, nb_channel))
    model = applications.VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    print(model.summary())
    features = dict()
    for product_id, img in dict_product_img.items():
        feature = model.predict(img, verbose=0)
        features[product_id] = feature

    return features


if prepare_img_data:
    print('Prepare image data ...')

    # key: product id, value: image
    dict_product_img = dict()

    for file_name in dirs:
        file_path = os.path.join(img_dir_path, file_name)

        # img = load_img(file_path)         # this is a PIL image
        img = load_img(file_path, target_size=(img_width, img_height))   # this is a PIL image
        x = img_to_array(img)                               # this is a Numpy array with shape (img_width, img_height, 3)
        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
        # x = x.reshape((1,) + x.shape)   # this is a Numpy array with shape (1, 3, img_width, img_height)
        # prepare the image for the VGG model
        x = preprocess_input(x)
        product_id = int(file_name.split('_')[0])

        if not int(product_id) in list_product_id:
            print('photo {} does not have product information'.format(file_path))
        else:
            dict_product_img[product_id] = x

    for product_id in list_product_id_df:
        if product_id not in dict_product_img:
            print('product {} does not have an image'.format(product_id))

    dict_product_img_features = extract_features(dict_product_img)
    # save features to file
    dump(dict_product_img_features, open(product_image_feature_file_path, 'wb'))

    print('save product image features to {}'.format(product_image_feature_file_path))


# --------------
# Prepare Text Data

product_description_file_path = 'product-descriptions.txt'

# https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
# Convert all words to lowercase.
# Remove all punctuation.
# Remove all words that are one character or less in length (e.g. ‘a’).
# Remove all words with numbers in them.
def clean_descriptions(descriptions):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc in descriptions.items():
        # tokenize
        desc = desc.split()
        # convert to lower case
        desc = [word.lower() for word in desc]
        # remove punctuation from each token
        desc = [w.translate(table) for w in desc]
        # remove hanging 's' and 'a'
        desc = [word for word in desc if len(word) > 1]
        # remove tokens with numbers in them
        desc = [word for word in desc if word.isalpha()]
        # store as string
        clean_str = ' '.join(desc)
        if not clean_str:
            print('cleaned description of product {} is empty'.format(key))
        else:
            descriptions[key] = clean_str

# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
    # build a list of all description strings
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc


# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc in descriptions.items():
        if not desc:
            print('product {} does not have a description'.format(key))
        # print(key)
        # print(desc)
        lines.append(str(key) + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# Create a dictionary, key: product id -> value: description
if prepare_text_data:
    print('Prepare text data ...')

    dict_product_des = dict()
    for product_id in list_product_id:
        # we assume that one product has only one description.
        if product_id in dict_product_des:
            print('product {} has more than one description'.format(product_id))
        description = df_product[df_product['ProductId']==product_id]['Description'].values[0]
        if not description:
            print('product {} does not have a description'.format(product_id))
        else:
            dict_product_des[product_id] = description

    for product_id in list_product_id[0:5]:
        print(dict_product_des[product_id])

    # clean descriptions
    clean_descriptions(dict_product_des)
    for product_id in list_product_id[0:5]:
        print(dict_product_des[product_id])

    # summarize vocabulary
    vocabulary = to_vocabulary(dict_product_des)
    print('Vocabulary Size: %d' % len(vocabulary))

    # save descriptions
    save_descriptions(dict_product_des, product_description_file_path)


# --------------
# Step 3: Develop deep learning model for text generation

# load doc into memory
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


# load clean descriptions into memory
def load_clean_descriptions(filename, list_product_id):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        product_id, product_desc = int(tokens[0]), tokens[1:]
        if product_id in list_product_id:
            if product_id not in descriptions:
                descriptions[product_id] = list()
            desc = 'startseq ' + ' '.join(product_desc) + ' endseq'
            descriptions[product_id].append(desc)

    # for key, value in descriptions.items():
    #     print(key)
    #     print(value)
    return descriptions


# load photo features
def load_photo_features(filename, list_product_id):
    # load all features
    all_features = load(open(filename, 'rb'))

    # features = {k: all_features[k] for k in list_product_id}

    # filter features
    dataset = []
    # dict_features = dict()
    for product_id in list_product_id:
        if (str(product_id) in all_features) or (product_id in all_features):
            dataset.append(product_id)

    # for product_id, features in all_features.items():
    #     if int(product_id) in list_product_id:
    #         dict_features[int(product_id)] = features

    # filter features
    features = {int(k): all_features[k] for k in dataset}

    return features


# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

# ---------
# encode the text
# For example, the input sequence “little girl running in field” would be split into 6 input-output pairs to train the model:
"""
X1,		X2 (text sequence), 						 y (word)
photo	startseq, 									 little
photo	startseq, little,							 girl
photo	startseq, little, girl, 					 running
photo	startseq, little, girl, running, 			 in
photo	startseq, little, girl, running, in, 		 field
photo	startseq, little, girl, running, in, field,  endseq
"""

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo):
    X1, X2, y = list(), list(), list()
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length):
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            # retrieve the photo feature
            if key in photos:
                photo = photos[key][0]
                in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
                yield [[in_img, in_seq], out_word]



# --------------
# Define the model

# define the captioning model
def define_model(vocab_size, max_length):

    # feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # summarize model
    print(model.summary())
    # plot_model(model, to_file='model.png', show_shapes=True)
    return model


print('Number of products: {}'.format(len(list_product_id)))

# image features
all_features = load_photo_features(product_image_feature_file_path, list_product_id)
print('All image features: %d' % len(all_features))

# descriptions
all_descriptions = load_clean_descriptions(product_description_file_path, list_product_id)
print('Descriptions: %d' % len(all_descriptions))

# print('Descriptions')
# for key, value in all_descriptions.items():
#     print(key)
#     print(value)

# prepare train and test sets
percentage_train = 0.8
list_train_product_id = list_product_id[0:int(len(list_product_id)*percentage_train)]
list_test_product_id = list_product_id[len(list_train_product_id):]

train_features = dict()
train_descriptions = dict()
for product_id in list_train_product_id:
    train_features[product_id] = all_features[product_id]
    train_descriptions[product_id] = all_descriptions[product_id]
# descriptions
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
print('Photos: train=%d' % len(train_features))

test_features = dict()
test_descriptions = dict()
for product_id in list_test_product_id:
    test_features[product_id] = all_features[product_id]
    test_descriptions[product_id] = all_descriptions[product_id]
# descriptions
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
print('Photos: test=%d' % len(test_features))


# prepare sequences
# X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)

# prepare tokenizer
tokenizer = create_tokenizer(all_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(all_descriptions)
print('Description Length: %d' % max_length)

# define the model
model = define_model(vocab_size, max_length)


if TRAIN_MODEL:
    print('Train model ... ')
    # train the model, run epochs manually and save after each epoch
    epochs = 2
    steps = len(train_descriptions)
    for i in range(epochs):
        # create the data generator
        generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
        # fit for one epoch
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        # save model
        model_path = 'model_' + str(i) + '.h5'
        model.save(model_path)
        print('save model to {}'.format(model_path))


# --------
# Evaluation
# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# prepare test set
if EVALUATE_MODEL:
    print('Evaluate model ...')
    # load the model
    # filename = 'model-ep002-loss3.245-val_loss3.612.h5'
    filename = 'model_1.h5'
    model = load_model(filename)
    # evaluate model
    evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)

