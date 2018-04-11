# Farfetech case study
#
# Product image generation with GANs
#
# Author: Kai Chen
# Date: Apr, 2018
#

# Reference
# https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0

import time
import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import load_img, img_to_array

import DCGAN

np.random.seed(42)

import matplotlib.pyplot as plt



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
img_width, img_height = 80, 80
img_dir_path = "data/images_{}_{}/".format(img_width, img_height)

dirs = os.listdir(img_dir_path)

for file_name in dirs:
    file_path = os.path.join(img_dir_path, file_name)
    product_id = int(file_name.split('_')[0])

    if not product_id in list_product_id_df:
        print('photo {} does not have product information'.format(file_path))
    else:
        list_product_id.append(product_id)

# key: product id, value: image
dict_product_img = dict()

for file_name in dirs:
    file_path = os.path.join(img_dir_path, file_name)

    # img = load_img(file_path)         # this is a PIL image
    img = load_img(file_path, target_size=(img_width, img_height))   # this is a PIL image
    x = img_to_array(img)                               # this is a Numpy array with shape (img_width, img_height, 3)
    # x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2]))

    # x = x.reshape((1,) + x.shape)   # this is a Numpy array with shape (1, 3, img_width, img_height)
    # prepare the image for the VGG model
    product_id = int(file_name.split('_')[0])

    if not int(product_id) in list_product_id:
        print('photo {} does not have product information'.format(file_path))
    else:
        dict_product_img[product_id] = x

# -----------
# Step 2: Define the model
class PRODUCT_DCGAN(object):
    def __init__(self, x_train):
        self.img_rows = 80
        self.img_cols = 80
        self.channel = 3

        # self.x_train = input_data.read_data_sets("mnist",\
        # 	one_hot=True).train.images
        self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, 1).astype(np.float32)

        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'product.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "product_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )


if __name__ == '__main__':
    x_train = []

    for product_id in dict_product_img:
        x_train.append(dict_product_img[product_id])

    product_dcgan = PRODUCT_DCGAN(x_train)

    timer = ElapsedTimer()

    # product_dcgan.train(train_steps=10000, batch_size=256, save_interval=500)
    product_dcgan.train(train_steps=10, batch_size=256, save_interval=5)

    timer.elapsed_time()
    product_dcgan.plot_images(fake=True)
    product_dcgan.plot_images(fake=False, save2file=True)
