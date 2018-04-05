import numpy as np
import pandas as pd


PRODUCT_CSV_FILE = 'data/products.csv'
ATTRIBUTE_CSV_FILE = 'data/attributes.csv'

df_product = pd.read_csv(PRODUCT_CSV_FILE)
print(df_product.head())
print(df_product.shape)
print(len(df_product['ProductId'].unique()))

df_attribute = pd.read_csv(ATTRIBUTE_CSV_FILE)
print(df_attribute.head())
print(df_attribute.shape)
print(len(df_attribute['ProductId'].unique()))

df_product_attribute = pd.merge(df_product, df_attribute, on=['ProductId'])
print(df_product_attribute.head())
print(df_product_attribute.shape)

print(len(df_product_attribute['ProductId'].unique()))
print(len(df_product_attribute['AttributeName'].unique()))
