#
# Farfetech case study
#
# Product category classification with CNN
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

np.random.seed(42)


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
from sklearn.metrics import classification_report

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ReduceLROnPlateau


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
list_product_id = np.array(list_product_id)
print('number of products {}'.format(list_product_id.shape[0]))

# get product category
print('get product category.')
list_category = df_product['Category'].unique()
list_category = np.array(list_category)
print('number of category {}'.format(list_category.shape[0]))

dict_product_cat = dict()
for product_id in list_product_id:
    category = df_product[df_product['ProductId'] == product_id]['Category'].values
    # if (len(category)) > 1:
    #     print('>1')
    dict_product_cat[product_id] = category[0]

# key: category  value: number of products
dict_cat = dict()
for category in list_category:
    products = df_product[df_product['Category'] == category]['ProductId'].values
    dict_cat[category] = products
    # print('category {}  number of products {}'.format(category, len(products)))




# get mapping product_id -> articlephoto_id
# ArticlePhotoId
list_photo_id = df_product['ArticlePhotoId'].unique()
dict_photo_product_id = dict()
for photo_id in list_photo_id:
    dict_photo_product_id[photo_id] = df_product[df_product['ArticlePhotoId']==photo_id]['ProductId'].values[0]


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
# Step 3: Train a CNN model for category classification

# category detection
class_names = list_category

# ideas are taken from
# https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
# https://machinelearningmastery.com/image-augmentation-deep-learning-keras/
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# https://github.com/tatsuyah/CNN-Image-Classifier/blob/master/src/train-multiclass.py
img_width, img_height = 100, 100

train_img_x = []
train_img_y = []
test_img_x = []
test_img_y = []

# ----
# get training and test images
img_dir_path = "data/images_{}_{}/".format(img_width, img_height)
dirs = os.listdir(img_dir_path)

for file_name in dirs:
    file_path = os.path.join(img_dir_path, file_name)

    img = load_img(file_path)      # this is a PIL image
    x = img_to_array(img)          # this is a Numpy array with shape (img_width, img_height, 3)
    # x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, img_width, img_height)

    product_id = int(file_name.split('_')[0])
    # print(product_id)

    # if not product_id in list_product_id:
    #     print('photo {} does not have product information'.format(photo_id))

    if product_id in list_product_id:
        if product_id in list_product_id_train:
            train_img_x.append(x)
            train_img_y.append(dict_product_cat[product_id])
        elif product_id in list_product_id_test:
            test_img_x.append(x)
            test_img_y.append(dict_product_cat[product_id])

train_img_x = np.array(train_img_x)
train_img_y = np.array(train_img_y)
test_img_x = np.array(test_img_x)
test_img_y = np.array(test_img_y)




# transform category to one-hot encoding
le = preprocessing.LabelEncoder()
le.fit(class_names)
train_img_y = le.transform(train_img_y)
test_img_y = le.transform(test_img_y)

# show number of samples per class
# plt.hist(train_img_y.tolist(), range(min(train_img_y), max(train_img_y)+1))
# plt.show()


train_img_y = to_categorical(train_img_y, num_classes = len(class_names))
test_img_y = to_categorical(test_img_y, num_classes = len(class_names))

# Split the train and the validation set for the fitting
train_img_x, val_img_x, train_img_y, val_img_y = train_test_split(train_img_x, train_img_y, test_size = 0.1, random_state=42)


# CNN hyperparameters
epochs = 1
batch_size = 128

model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = (5,5), padding = 'Same', activation ='relu', input_shape = (img_width, img_height, 3)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
#                  activation ='relu'))
# model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
#                  activation ='relu'))
# model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(len(class_names), activation = "softmax"))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

# -----------
# Data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,               # set input mean to 0 over the dataset
        samplewise_center=False,                # set each sample mean to 0
        featurewise_std_normalization=False,    # divide inputs by std of the dataset
        samplewise_std_normalization=False,     # divide each input by its std
        zca_whitening=False,                    # apply ZCA whitening
        rotation_range=10,                      # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1,                       # randomly zoom image
        width_shift_range=0.1,                  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,                 # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,                  # randomly flip images
        vertical_flip=False)                    # randomly flip images

datagen.fit(train_img_x)

# Fit the model
history = model.fit_generator(datagen.flow(train_img_x, train_img_y, batch_size=batch_size),
                              epochs = epochs, validation_data = (val_img_x, val_img_y),
                              verbose = 2, steps_per_epoch=train_img_x.shape[0] // batch_size, callbacks=[learning_rate_reduction])


# -----------
# Evaluation

# Training and validation curves
# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

plt.show()

# Confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
test_pred = model.predict(test_img_x)

# Convert predictions classes to one hot vectors
test_pred_classes = np.argmax(test_pred, axis = 1)
# results = pd.Series(results, name="Label")

# Convert test observations to one hot vectors
test_true_classes = np.argmax(test_img_y, axis = 1)

# compute the confusion matrix
test_true_classes = le.inverse_transform(test_true_classes)
test_pred_classes = le.inverse_transform(test_pred_classes)

print(classification_report(test_true_classes, test_pred_classes, target_names=class_names))

confusion_mtx = confusion_matrix(test_true_classes, test_pred_classes)
print(confusion_mtx)

# plot the confusion matrix
# plot_confusion_matrix(confusion_mtx, classes = range(10))
# plt.show()


# ----------
# TODO: Display some error results
test_true_classes = np.array(test_true_classes).reshape((test_img_x.shape[0], 1))
test_pred_classes = np.array(test_pred_classes).reshape((test_img_x.shape[0], 1))


error_indices = []
for i, val in enumerate(test_true_classes):
    if test_true_classes[i][0] != test_pred_classes[i][0]:
        error_indices.append(i)

list_error_product_id = []
for i in error_indices:
    list_error_product_id.append(list_product_id_test[i])

print('error classified products')
print(list_error_product_id)




# Errors are difference between predicted labels and true labels
# errors = (test_pred_classes - test_true != 0)
#
# Y_pred_classes_errors = test_pred_classes[errors]
# Y_pred_errors = test_pred[errors]
# Y_true_errors = test_true[errors]
# X_val_errors = test_img_x[errors]
#
# def display_errors(errors_index, img_errors, pred_errors, obs_errors, img_width, img_height):
#     """ This function shows 6 images with their predicted and real labels"""
#     n = 0
#     nrows = 2
#     ncols = 3
#     fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
#     for row in range(nrows):
#         for col in range(ncols):
#             error = errors_index[n]
#             ax[row,col].imshow((img_errors[error]).reshape((img_width, img_height)))
#             ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
#             n += 1
#
# # Probabilities of the wrong predicted numbers
# Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)
#
# # Predicted probabilities of the true values in the error set
# true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
#
# # Difference between the probability of the predicted label and the true label
# delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
#
# # Sorted list of the delta prob errors
# sorted_dela_errors = np.argsort(delta_pred_true_errors)
#
# # Top 6 errors
# most_important_errors = sorted_dela_errors[-6:]
#
# # Show the top 6 errors
# display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors, img_width, img_height)


# # predict results
# results = model.predict(test)
#
# # select the indix with the maximum probability
# results = np.argmax(results,axis = 1)
#
# results = pd.Series(results,name="Label")



# # -----------
# # Step 4: NLP
# # remove stop words
# df_attribute = pd.read_csv(ATTRIBUTE_CSV_FILE)
# print(df_attribute.head())
# print(df_attribute.shape)
# print(len(df_attribute['ProductId'].unique()))
#
# df_product_attribute = pd.merge(df_product, df_attribute, on=['ProductId'])
# print(df_product_attribute.head())
# print(df_product_attribute.shape)
#
# print(len(df_product_attribute['ProductId'].unique()))
# print(len(df_product_attribute['AttributeValueName'].unique()))
#
# print(df_product_attribute.columns)
# print(df_product_attribute.describe())
#
# list_product_id = df_attribute['ProductId'].unique()
# dict_product_des = dict()
#
#
# print('get product description.')
# for product_id in list_product_id:
#     if product_id in dict_product_des:
#         print('product {} has more than one description'.format(product_id))
#     df_sub = df_product_attribute[df_product_attribute['ProductId'] == product_id]
#     # print(df_sub['Description'].values[0])
#     dict_product_des[product_id] = df_sub['Description'].values[0]
#
# # create one-hot encoding for the attribute name
# print('get product attribute.')
# list_attribute = df_attribute['AttributeName'].unique()
# dict_product_att = dict()
# for product_id in list_product_id:
#     dict_product_att[product_id] = dict()
#     for attribute in list_attribute:
#         dict_product_att[product_id][attribute] = 0
#     list_product_attribute = df_product_attribute[df_product_attribute['ProductId'] == product_id]['AttributeName']
#     # print(len(list_product_attribute))
#     for attribute in list_product_attribute:
#         dict_product_att[product_id][attribute] = 1
#
# nlp = False
#
# def cleanupDoc(s):
#     stopset = set(stopwords.words('english'))
#     stopset.add('wikipedia')
#     tokens = text_to_word_sequence(s, filters="\"!'#$%&()*+,-˚˙./:;‘“<=·>?@[]^_`{|}~\t\n", lower=True, split=" ")
#     cleanup = " ".join(filter(lambda word: word not in stopset, tokens))
#     return cleanup
#
# if nlp:
#     train_text = []
#     train_attribute = dict()
#
#     for class_name in class_names:
#         train_attribute[class_name] = []
#
#     for product_id in list_product_id_train:
#         train_text.append(dict_product_des[product_id])
#         for class_name in class_names:
#             train_attribute[class_name].append(dict_product_att[product_id][class_name])
#
#     test_text = []
#     test_attribute = dict()
#
#     for class_name in class_names:
#         test_attribute[class_name] = []
#
#     for product_id in list_product_id_test:
#         test_text.append(dict_product_des[product_id])
#         for class_name in class_names:
#             test_attribute[class_name].append(dict_product_att[product_id][class_name])
#
#     train_text = [cleanupDoc(text) for text in train_text]
#     test_text = [cleanupDoc(text) for text in test_text]
#
#
#     word_vectorizer = TfidfVectorizer(
#         sublinear_tf=True,
#         strip_accents='unicode',
#         analyzer='word',
#         token_pattern=r'\w{1,}',
#         ngram_range=(1, 2),
#         # max_features=50000,
#         max_features=10000,
#         )
#     train_word_features = word_vectorizer.fit_transform(train_text)
#     print('Word TFIDF 1/2')
#     test_word_features = word_vectorizer.transform(test_text)
#     print('Word TFIDF 2/2')
#
#     char_vectorizer = TfidfVectorizer(
#         sublinear_tf=True,
#         strip_accents='unicode',
#         analyzer='char',
#         stop_words='english',
#         ngram_range=(2, 6),
#         # max_features=50000,
#         max_features=10000,
#         )
#     train_char_features = char_vectorizer.fit_transform(train_text)
#     print('Char TFIDF 1/2')
#     test_char_features = char_vectorizer.transform(test_text)
#     print('Char TFIDF 2/2')
#
#     train_features = hstack([train_char_features, train_word_features])
#     print('HStack 1/2')
#     test_features = hstack([test_char_features, test_word_features])
#     print('HStack 2/2')
#
#     pred_attribute = dict()
#     scores = []
#     for class_name in class_names:
#         # print(class_name)
#         train_target = train_attribute[class_name]
#
#         classifier = LogisticRegression(solver='sag')
#         sfm = SelectFromModel(classifier, threshold=0.2)
#
#         # print(train_features.shape)
#         train_sparse_matrix = sfm.fit_transform(train_features, train_target)
#         # print(train_sparse_matrix.shape)
#
#         # train_sparse_matrix, valid_sparse_matrix, y_train, y_valid = train_test_split(train_sparse_matrix, train_target,
#         #                                                                               test_size=0.05, random_state=42)
#         test_sparse_matrix = sfm.transform(test_features)
#
#         cv_score = np.mean(cross_val_score(classifier, train_sparse_matrix, train_target, cv=2, scoring='roc_auc'))
#         scores.append(cv_score)
#         # print('CV score for class {} is {}'.format(class_name, cv_score))
#
#         classifier.fit(train_sparse_matrix, train_target)
#
#         pred_attribute[class_name] = classifier.predict_proba(test_sparse_matrix)[:, 1]
#
#     # --------------
#     # evaluate the NLP model





# ---------
# Future work
