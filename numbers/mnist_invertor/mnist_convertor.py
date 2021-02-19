import gzip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import png

# read the train and test data and labels
with gzip.open("MNIST_gzip/train-images-idx3-ubyte.gz",'r') as f:
    f.read(16)
    x_train = np.frombuffer(f.read(60000*28*28), dtype=np.uint8).astype(np.uint8)
    x_train = x_train.reshape(60000,28,28,1)
with gzip.open("MNIST_gzip/train-labels-idx1-ubyte.gz",'r') as f:
    f.read(8)
    y_train = np.frombuffer(f.read(60000), dtype=np.uint8).astype(np.int64)
with gzip.open("MNIST_gzip/t10k-images-idx3-ubyte.gz",'r') as f:
    f.read(16)
    x_test = np.frombuffer(f.read(10000*28*28), dtype=np.uint8).astype(np.uint8)
    x_test = x_test.reshape(10000,28,28,1)
with gzip.open("MNIST_gzip/t10k-labels-idx1-ubyte.gz",'r') as f:
    f.read(8)
    y_test = np.frombuffer(f.read(10000), dtype=np.uint8).astype(np.int64)

print("Done loading!")

# save the normal images accordingly
for i in range(len(x_train)):
    png.from_array(x_train[i], 'L').save(f"MNIST_Data/normal/{y_train[i]}/train{i}.png")
print(f"{i+1} images done as part of training set!")

for i in range(len(x_test)):
    png.from_array(x_test[i], 'L').save(f"MNIST_Data/normal/{y_test[i]}/test{i}.png")
print(f"{i+1} images done as part of test set!")


# save the inverted images accordingly
for i in range(len(x_train)):
    png.from_array(255-x_train[i], 'L').save(f"MNIST_Data/inverted/{y_train[i]}/train{i}.png")
print(f"{i+1} inverted images done as part of training set!")

for i in range(len(x_test)):
    png.from_array(255-x_test[i], 'L').save(f"MNIST_Data/inverted/{y_test[i]}/test{i}.png")
print(f"{i+1} inverted images done as part of test set!")

