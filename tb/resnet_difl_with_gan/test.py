import os
import sys
import time
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Normalization, Rescaling, CenterCrop
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, roc_auc_score, auc

img_height, img_width = 200,200
secondary_img_height, secondary_img_width = 8,8 


# function to refresh a dataset
def refresh_dataset(dataset):
    print("Refreshing!")
    dataset = dataset.unbatch()
    dataset = dataset.shuffle(50, reshuffle_each_iteration=False)
    dataset = dataset.batch(2)
    print("In function")
    iterate(dataset)
    return dataset

def iterate(dataset):
    for i in dataset:
        print(i)

test = tf.data.Dataset.range(20)
test = test.batch(2)

iterate(test)

test = refresh_dataset(test)
print("After solo refresh")

iterate(test)

print("")
iterator = iter(test)
for i in range(100):
    try:
        x = iterator.get_next()
    except:
        iterator = iter(refresh_dataset(test))
        time.sleep(5)
        print("Out of function")
        x = iterator.get_next()
    print(x)

print("\nDone!")
