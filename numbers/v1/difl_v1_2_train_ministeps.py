import os
import sys
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Normalization, Rescaling, CenterCrop
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, roc_auc_score, auc

# function to refresh a dataset
def refresh_dataset(dataset):
    dataset = dataset.unbatch()
    dataset = dataset.shuffle(100000)
    dataset = dataset.batch(16)
    return dataset

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
Preparing the difl model
'''

# define input layer
inputs = keras.Input(shape=(28,28,1))

# conduct rescaling of features
x = Rescaling(scale=1.0/255)(inputs)

# apply convolution and pooling layers
x = layers.Conv2D(32, (3,3), activation = 'relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, (3,3), activation = 'relu')(x)
x = layers.MaxPooling2D()(x)
outputs = layers.Conv2D(128, (3,3), activation = 'relu')(x)

# define the final model
difl_model = keras.Model(inputs=inputs, outputs=outputs, name="DIFL_Model")

# display the difl model summary
print(difl_model.summary())


'''
Preparing the classification model
'''

# define input layer
inputs1 = keras.Input(shape = (3,3,128))

# apply convolution and pooling layers
x = layers.Flatten()(inputs1)
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
mse_loss = keras.losses.MeanSquaredError()

# instantiate the accuracy metrics
train_accuracy = keras.metrics.SparseCategoricalAccuracy()
test_accuracy = keras.metrics.SparseCategoricalAccuracy()

'''
Start training the model as well as its evaluation
'''
source_iterator = iter(source_train)
target_iterator = iter(target_train)

for epoch in range(epochs):
    print(f"Starting epoch {epoch+1}/{epochs} of DIFL training!")

    # custom training loop for each batch in the training dataset
    for i in range(len(source_train)):
        
        # get the next batch from the source dataset
        try:
            xsourcebatch, ysourcebatch = source_iterator.get_next()
        except:
            source_iterator = iter(refresh_dataset(source_train))
            xsourcebatch, ysourcebatch = source_iterator.get_next()
        
        # get the next batch from the target dataset
        try:
            xtargetbatch, ytargetbatch = target_iterator.get_next()
        except:
            target_iterator = iter(refresh_dataset(target_train))
            xtargetbatch, ytargetbatch = target_iterator.get_next()
        
        # DIFL learning step
        with tf.GradientTape() as tape:
            logits1 = difl_model(xsourcebatch, training=True)
            logits2 = difl_model(xtargetbatch, training=True)
            difl_loss = mse_loss(logits1, logits2)
        gradients1 = tape.gradient(difl_loss, difl_model.trainable_weights)
        optimizer.apply_gradients(zip(gradients1, difl_model.trainable_weights))
        
        # classification learning step
        with tf.GradientTape(persistent=True) as tape:
            temp = difl_model(xsourcebatch, training=True)
            logits3 = classification_model(temp, training=True)
            classification_loss = categorical_loss(ysourcebatch, logits3)
        gradients2 = tape.gradient(classification_loss, difl_model.trainable_weights)
        gradients3 = tape.gradient(classification_loss, classification_model.trainable_weights)
        del tape

        # apply the gradients
        optimizer.apply_gradients(zip(gradients2, difl_model.trainable_weights))
        optimizer.apply_gradients(zip(gradients3, classification_model.trainable_weights))
        # optimizer.apply_gradients(zip(gradients1, difl_model.trainable_weights))

        
        train_accuracy.update_state(ysourcebatch, logits3)
        print(f"The training accuracy at batch {i+1}/{len(source_train)} is: {float(train_accuracy.result())}.")
    
    # reset the training accuracy at the end of each epoch
    train_accuracy.reset_states()


# test the model on normal MNIST train
for xbatch, ybatch in source_train:
    encodings = difl_model(xbatch, training=False)
    logits = classification_model(encodings, training=False)
    test_accuracy.update_state(ybatch, logits)
normal_train = float(test_accuracy.result())
print(f"The accuracy on the training set of MNIST is: {float(test_accuracy.result())}.")
test_accuracy.reset_states()

# test the model on normal MNIST test
for xbatch, ybatch in source_test:
    encodings = difl_model(xbatch, training=False)
    logits = classification_model(encodings, training=False)
    test_accuracy.update_state(ybatch, logits)
normal_test = float(test_accuracy.result())
print(f"The accuracy on the test set of MNIST is: {float(test_accuracy.result())}.")
test_accuracy.reset_states()

# test the model on inverted MNIST train
for xbatch, ybatch in target_train:
    encodings = difl_model(xbatch, training=False)
    logits = classification_model(encodings, training=False)
    test_accuracy.update_state(ybatch, logits)
inverted_train = float(test_accuracy.result())
print(f"The accuracy on the training set of Inverted MNIST is: {float(test_accuracy.result())}.")
test_accuracy.reset_states()

# test the model on inverted MNIST test
for xbatch, ybatch in target_test:
    encodings = difl_model(xbatch, training=False)
    logits = classification_model(encodings, training=False)
    test_accuracy.update_state(ybatch, logits)
inverted_test = float(test_accuracy.result())
print(f"The accuracy on the test set of Inverted MNIST is: {float(test_accuracy.result())}.")
test_accuracy.reset_states()
        
print("\nDone!")
print(f"{normal_train};{normal_test};{inverted_train};{inverted_test}")
