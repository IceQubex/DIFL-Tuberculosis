import os
import sys
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Normalization, Rescaling, CenterCrop
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, roc_auc_score, auc

# parse input arguments into lists
source_directory = sys.argv[1]
target_directory = sys.argv[2]
epochs = int(sys.argv[3])

print(f"\nImporting the necessary datasets..")
# import the source dataset
source_train = keras.preprocessing.image_dataset_from_directory(source_directory, labels='inferred', label_mode='int', color_mode='grayscale', batch_size=16, image_size=(28,28), validation_split=0.2, subset='training', seed=123)
source_test = keras.preprocessing.image_dataset_from_directory(source_directory, labels='inferred', label_mode='int', color_mode='grayscale', batch_size=16, image_size=(28,28), validation_split=0.2, subset='validation', seed=123)

# import the target dataset
target_train = keras.preprocessing.image_dataset_from_directory(target_directory, labels='inferred', label_mode='int', color_mode='grayscale', batch_size=16, image_size=(28,28), validation_split=0.2, subset='training', seed=132)
target_test = keras.preprocessing.image_dataset_from_directory(target_directory, labels='inferred', label_mode='int', color_mode='grayscale', batch_size=16, image_size=(28,28), validation_split=0.2, subset='validation', seed=132)

print("Datasets imported!")

'''
Preparing the classification model
'''

# define input layer
inputs1 = keras.Input(shape = (28,28,1))

# conduct rescaling of features
x = Rescaling(scale=1.0/255)(inputs1)

# apply convolution and pooling layers
x = layers.Conv2D(32,(3,3), activation = 'relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64,(3,3), activation = 'relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, (3,3), activation = 'relu')(x)

# flatten into dense layers
x = layers.Flatten()(x)
x = layers.Dense(128, activation = 'relu')(x)
x = layers.Dense(64, activation = 'relu')(x)

# define output layer
outputs1 = layers.Dense(10, activation = 'softmax')(x)

# define the final model
classification_model = keras.Model(inputs=inputs1, outputs=outputs1, name = "Classification_Model")

# display the model summary
print(classification_model.summary())


'''
Specificy the other parameters for the model
'''

# instantiate the optimizer
optimizer = keras.optimizers.Adam(lr=0.001)

# instantiate the loss function
categorical_loss = keras.losses.SparseCategoricalCrossentropy()

# instantiate the accuracy metrics
train_accuracy = keras.metrics.SparseCategoricalAccuracy()
test_accuracy = keras.metrics.SparseCategoricalAccuracy()

'''
Start training the model as well as its evaluation
'''

# start the training iteration
for epoch in range(epochs):
    print(f"Starting epoch {epoch+1}/{epochs} of DIFL learning!")
    
    # custom training loop for each batch of the training dataset
    for i, (xbatch, ybatch) in enumerate(source_train):
        ybatch = tf.dtypes.cast(ybatch, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(xbatch)
            tape.watch(ybatch)
            logits = classification_model(xbatch, training=True)
            train_loss = categorical_loss(ybatch, logits)
        gradients = tape.gradient(train_loss, classification_model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, classification_model.trainable_weights))
        train_accuracy.update_state(ybatch, logits)
        #print(f"The current training accuracy at batch {i+1}/{len(source_train)} is: {float(train_accuracy.result())}.")

    # reset the accuracy metrics at the end of the epoch
    train_accuracy.reset_states()

# calculate accuracy on normal MNIST train
for xbatch,ybatch in source_train:
    logits = classification_model(xbatch, training=False)
    test_accuracy.update_state(ybatch, logits)
normal_train = float(test_accuracy.result())
print(f"\nThe accuracy on the training set of normal MNIST is: {float(test_accuracy.result())}.")
test_accuracy.reset_states()

# calculate accuracy on normal MNIST test
for xbatch,ybatch in source_test:
    logits = classification_model(xbatch, training=False)
    test_accuracy.update_state(ybatch, logits)
normal_test = float(test_accuracy.result())
print(f"The accuracy on the testing set of normal MNIST is: {float(test_accuracy.result())}.")
test_accuracy.reset_states()

# calculate accuracy on the inverted MNIST train
for xbatch, ybatch in target_train:
    logits = classification_model(xbatch, training=False)
    test_accuracy.update_state(ybatch, logits)
inverted_train = float(test_accuracy.result())
print(f"The accuracy on the training set of inverted MNIST is: {float(test_accuracy.result())}.")
test_accuracy.reset_states()

# calculate accuracy on the inverted MNIST test
for xbatch, ybatch in target_test:
    logits = classification_model(xbatch, training=False)
    test_accuracy.update_state(ybatch, logits)
inverted_test = float(test_accuracy.result())
print(f"The accuracy on the test set of inverted MNIST is: {float(test_accuracy.result())}.")
        
print("\nDone!")
print(f"{normal_train};{normal_test};{inverted_train};{inverted_test}")
