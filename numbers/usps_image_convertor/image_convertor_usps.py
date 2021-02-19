import gzip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import png
import h5py

# read the train and test data and labels
with h5py.File("usps.h5", 'r') as hf:
    train = hf.get('train')
    x_train = train.get('data')[:]
    y_train = train.get('target')[:]
    test = hf.get('test')
    x_test = test.get('data')[:]
    y_test = test.get('target')[:]

print("Done loading!")
x_train = x_train*255
x_train = x_train.astype(np.uint8)
x_test = x_test*255
x_test = x_test.astype(np.uint8)

# save the images accordingly
for i in range(len(x_train)):
    png.from_array(x_train[i].reshape((16,16)), 'L').save(f"USPS_Data/train/{y_train[i]}/train{i}.png")
print(f"{i+1} images done as part of training set!")

for i in range(len(x_test)):
    png.from_array(x_test[i].reshape((16,16)), 'L').save(f"USPS_Data/test/{y_test[i]}/test{i}.png")
print(f"{i+1} images done as part of test set!")
'''
for i in range(len(x_train)):
    print(x_train[i])
    time.sleep(1)
'''
