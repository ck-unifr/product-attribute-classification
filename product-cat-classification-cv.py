# Farfetech case study
#
# Product category classification with CNN
# The object of this script is using deep learning technologies (CNN) for product category classification
#
# Author: Kai Chen
# Date: Apr, 2018
#


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

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras import callbacks
from keras import applications
from keras import optimizers

np.random.seed(42)

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

list_product_id = df_product['ProductId'].unique()
list_product_id = np.array(list_product_id)
print('number of products {}'.format(list_product_id.shape[0]))

list_category = df_product['Category'].unique()
list_category = np.array(list_category)
print('number of category {}'.format(list_category.shape[0]))

# create a dictionary with key: product id -> value: category
# in this csv file we can find that each product belongs only to one category
dict_product_cat = dict()
for product_id in list_product_id:
    category = df_product[df_product['ProductId'] == product_id]['Category'].values
    if (len(category)) > 1:
         print('product {} belongs to more than two categories.'.format(product_id))
    dict_product_cat[product_id] = category[0]

# create a dictionary with key: category -> value: a list of product id
dict_cat = dict()
dict_cat_nb_products = dict()
nb_products_cat = []
for category in list_category:
    products = df_product[df_product['Category'] == category]['ProductId'].values
    dict_cat[category] = products
    dict_cat_nb_products[category] = len(products)
    nb_products_cat.append(len(products))
    # print('category {}  number of products {}'.format(category, len(products)))


# show number of products per category
print('plot number of products per category')
plt.bar(range(len(dict_cat_nb_products)), list(dict_cat_nb_products.values()), align='center')
plt.xticks(range(len(dict_cat_nb_products)), list(dict_cat_nb_products.keys()))
plt.xticks(rotation=90)
# # for python 2.x:
# plt.bar(range(len(dict_cat_nb_products)), dict_cat_nb_products.values(), align='center')  # python 2.x
# plt.xticks(range(len(dict_cat_nb_products)), dict_cat_nb_products.keys())  # in python 2.x
if plot_figure:
    plt.show()

# get max and min number of products per category
min_products = sys.maxsize
min_products_cat = ''
max_products = 0
max_products_cat = ''
for category, nb_products in dict_cat_nb_products.items():
    if nb_products < min_products:
        min_products = nb_products
        min_products_cat = category
    if nb_products > max_products:
        max_products = nb_products
        max_products_cat = category

print('category {} has the max number of products, i.e., {}'.format(max_products_cat, max_products))
print('category {} has the min number of products, i.e., {}'.format(min_products_cat, min_products))
print('mean number of products per category: {}'.format(np.mean(nb_products_cat)))
print('standard deviation of number of products per category: {}'.format(np.std(nb_products_cat)))


# create a dictionary with key: photo id -> value: product id
# note one photo belongs only to one product
list_photo_id = df_product['ArticlePhotoId'].unique()
dict_photo_product_id = dict()
for photo_id in list_photo_id:
    dict_photo_product_id[photo_id] = df_product[df_product['ArticlePhotoId']==photo_id]['ProductId'].values[0]


# ---------
# Step 2: Prepare train, and test sets

shuffle(list_product_id)
percentage_train_set = 0.7
list_product_id_train = list_product_id[0:int(percentage_train_set*len(list_product_id))]
list_product_id_test = list_product_id[len(list_product_id_train):]

print('number of samples {}'.format(len(list_product_id)))
print('number of trainig samples {}'.format(len(list_product_id_train)))
print('number of trainig samples {}'.format(len(list_product_id_test)))



# -----------
# Step 3: Train a CNN model for category classification

# The ideas are taken from:
# https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
# https://machinelearningmastery.com/image-augmentation-deep-learning-keras/
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# https://github.com/tatsuyah/CNN-Image-Classifier/blob/master/src/train-multiclass.py
# https://www.kaggle.com/fujisan/use-keras-pre-trained-vgg16-acc-98


class_names = list_category

img_width, img_height = 80, 80
#img_width, img_height = 100, 100

train_img_x = []
train_img_y = []
test_img_x = []
test_img_y = []

# get training and test images
img_dir_path = "data/images_{}_{}/".format(img_width, img_height)
dirs = os.listdir(img_dir_path)

# key: product id, value: image path
dict_img_path = dict()

for file_name in dirs:
    file_path = os.path.join(img_dir_path, file_name)

    img = load_img(file_path)         # this is a PIL image
    x = img_to_array(img)             # this is a Numpy array with shape (img_width, img_height, 3)
    # x = x.reshape((1,) + x.shape)   # this is a Numpy array with shape (1, 3, img_width, img_height)

    product_id = int(file_name.split('_')[0])

    dict_img_path[product_id] = file_path

    if not product_id in list_product_id:
        print('photo {} does not have product information'.format(file_path))

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


# show number of samples per class in the train set
plt.hist(train_img_y.tolist(), range(min(train_img_y), max(train_img_y)+1))
if plot_figure:
    plt.show()


train_img_y = to_categorical(train_img_y, num_classes = len(class_names))
test_img_y = to_categorical(test_img_y, num_classes = len(class_names))


# split the train and the validation set for the fitting
train_img_x, val_img_x, train_img_y, val_img_y = train_test_split(train_img_x, train_img_y, test_size = 0.1, random_state=42)


# ----------------------
# CNN hyperparameters

epochs = 2
batch_size = 32
filters = [16, 16, 8, 8]
kernel_sizes = [11, 11, 7, 7]
strides = [2, 2, 2, 2]
pooling_sizes = [2, 2]
str_parameters = '[epochs]{}-[batch_size]{}-[filters]{}-[kernel_sizes]{}-[strides]{}-[pooling_sizes]{}'.format(epochs,
                                                                                                                batch_size,
                                                                                                                '_'.join(str(x) for x in filters),
                                                                                                                '_'.join(str(x) for x in kernel_sizes),
                                                                                                                '_'.join(str(x) for x in strides),
                                                                                                                '_'.join(str(x) for x in pooling_sizes),
                                                                                                                )

#model_name = 'CNN'
model_name = 'VGG16'

model = None
if model_name == 'CNN':
    model = Sequential()

    model.add(Conv2D(filters = filters[0], kernel_size = (kernel_sizes[0], kernel_sizes[0]),
                     padding = 'Same', strides=strides[0],  input_shape = (img_width, img_height, 3)),
                     #activation ='relu',
                    )
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters = filters[1], kernel_size = (kernel_sizes[1], kernel_sizes[1]),
                     padding = 'Same', strides=strides[1],
                     #activation ='relu'
                     ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=(pooling_sizes[0], pooling_sizes[0])))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters = filters[2], kernel_size = (kernel_sizes[2], kernel_sizes[2]),
                     padding = 'Same', strides=strides[2],
                     #activation ='relu'
                     ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters = filters[3], kernel_size = (kernel_sizes[3], kernel_sizes[3]),
                     padding = 'Same', strides=strides[3],
                     #activation ='relu'
                     ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=(pooling_sizes[1], pooling_sizes[1])))
    model.add(Dropout(0.2))

    model.add(Flatten())
    #model.add(Dense(256, activation = "relu"))
    model.add(Dense(256, kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(len(class_names), activation = "softmax"))

elif model_name == 'VGG16':
    # use pre-trained VGG16
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))

    add_model.add(Dense(256, kernel_initializer='glorot_uniform'))
    # add_model.add(Dense(1, activation='sigmoid'))
    add_model.add(BatchNormalization())
    add_model.add(Activation('relu'))
    add_model.add(Dropout(0.2))

    add_model.add(Dense(len(class_names), activation="softmax"))

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

    # model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    #              metrics=['accuracy'])

print(model.summary())

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

# -----------
# data augmentation
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
# Tensorboard log
tf_log_dir = './tf-log/'
if not os.path.exists(tf_log_dir):
    os.makedirs(tf_log_dir)
tb_cb = callbacks.TensorBoard(log_dir=tf_log_dir, histogram_freq=1)
cbks = [tb_cb]

history = model.fit_generator(datagen.flow(train_img_x, train_img_y, batch_size=batch_size),
                              epochs = epochs, validation_data = (val_img_x, val_img_y),
                              verbose = 2, steps_per_epoch=train_img_x.shape[0] // batch_size, callbacks=[learning_rate_reduction])


# Training and validation curves
# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

if plot_figure:
    plt.show()

# save the model
model_dir = './models/'
if not os.path.exists(model_dir):
  os.mkdir(model_dir)

model_path = ''
if model_name == 'CNN':
    model_path = '{}cat-cnn-model-{}.h5'.format(model_dir, str_parameters)
elif model_name == 'VGG16':
    model_path = '{}cat-vgg16-{}-{}-{}.h5'.format(model_dir, epochs, batch_size)
model.save(model_path)
print('save model to {}'.format(model_path))

model_weights_path = ''
if model_name == 'CNN':
    model_weights_path = '{}cat-cnn-weights-{}.h5'.format(model_dir, str_parameters)
elif model_name == 'VGG16':
    model_weights_path = '{}cat-vgg16-weights-{}-{}.h5'.format(model_dir, epochs, batch_size)
model.save_weights(model_weights_path)
print('save weights to {}'.format(model_weights_path))


# -----------
# Step 4: Make predictions

# load the model
model = load_model(model_path)
model.load_weights(model_weights_path)

test_pred = model.predict(test_img_x)


# -----------
# Step 5: Evaluation

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

# convert predictions classes to one hot vectors
test_pred_classes = np.argmax(test_pred, axis = 1)
# results = pd.Series(test_pred_classes, name="Label")

# convert test observations to one hot vectors
test_true_classes = np.argmax(test_img_y, axis = 1)

# compute the confusion matrix
test_true_classes = le.inverse_transform(test_true_classes)
test_pred_classes = le.inverse_transform(test_pred_classes)

print(classification_report(test_true_classes, test_pred_classes, target_names=class_names))

confusion_mtx = confusion_matrix(test_true_classes, test_pred_classes)
print(confusion_mtx)

# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10))

if plot_figure:
    plt.show()


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

acc = 1 - len(list_error_product_id) / len(list_product_id_test)
print('accuracy {}'.format(acc))

img = mpimg.imread(dict_img_path[list_error_product_id[0]])
imgplot = plt.imshow(img)

if plot_figure:
    plt.show()





# ---------
# Future work
