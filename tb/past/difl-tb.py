import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Normalization, Rescaling, CenterCrop
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, roc_auc_score, auc

def refresh_dataset(dataset):
    dataset = dataset.unbatch()
    dataset = dataset.shuffle(100000)
    dataset = dataset.batch()


# parse input arguments into lists
source_directory = sys.argv[1]
target_directory = sys.argv[2]
epochs = int(sys.argv[3])

print(f"\nImporting the necessary datasets..")
# import the source dataset
source_train = keras.preprocessing.image_dataset_from_directory(source_directory, labels='inferred', label_mode='binary', color_mode='grayscale', batch_size=1, image_size=(512,512), validation_split=0.2, subset='training', seed=123)
source_test = keras.preprocessing.image_dataset_from_directory(source_directory, labels='inferred', label_mode='binary', color_mode='grayscale', batch_size=1, image_size=(512,512), validation_split=0.2, subset='validation', seed=123)

# import the target dataset
target_train = keras.preprocessing.image_dataset_from_directory(target_directory, labels='inferred', label_mode='binary', color_mode='grayscale', batch_size=1, image_size=(512,512), validation_split=0.2, subset='training', seed=123)
target_test = keras.preprocessing.image_dataset_from_directory(target_directory, labels='inferred', label_mode='binary', color_mode='grayscale', batch_size=1, image_size=(512,512), validation_split=0.2, subset='validation', seed=123)

print("Datasets imported!")


'''
Preparing the difl model
'''

# define input layer
inputs = keras.Input(shape=(512,512,1))

# conduct rescaling of features
x = Rescaling(scale=1.0/255)(inputs)

# apply convolution and pooling layers
x = layers.Conv2D(32, (7,7), activation = 'relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, (5,5), activation = 'relu')(x)
x = layers.MaxPooling2D()(x)
outputs = layers.Conv2D(128, (5,5), activation = 'relu')(x)

# define the final model
difl_model = keras.Model(inputs=inputs, outputs=outputs, name="DIFL_Model")

# display the difl model summary
print(difl_model.summary())


'''
Preparing the classification model
'''

# define input layer
inputs1 = keras.Input(shape = (120,120,128))

# apply convolution and pooling layers
x = layers.Conv2D(128, (5,5), activation = 'relu')(inputs1)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, (3,3), activation = 'relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, (3,3), activation = 'relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation = 'relu')(x)
x = layers.Dense(64, activation = 'relu')(x)

# define output layer
outputs1 = layers.Dense(1, activation = 'sigmoid')(x)

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
binary_loss = keras.losses.BinaryCrossentropy()
mse_loss = keras.losses.MeanSquaredError()

# instantiate the accuracy metrics
train_accuracy = keras.metrics.BinaryAccuracy()
test_accuracy = keras.metrics.BinaryAccuracy()

'''
Start training the model as well as its evaluation
'''
source_iterator = iter(source_train)
target_iterator = iter(test_train) 

for epoch in range(epochs):
    print(f"Starting epoch {epoch+1}/{epochs} of DIFL training!")

    # custom training loop for each batch in the training dataset
    while True:

        xsourcebatch, ysourcebatch = source_iterator.get_next()
        xtargetbatch, ytargetbatch = target_iterator.get_next()
        if len(xtargetbatch) != 16:
            

        
        # DIFL network training step
        with tf.GradientTape() as tape:
            logits1 = difl_model(xsourcebatch, training=True)
            logits2 = difl_model(xtargetbatch, training=True)
            difl_loss = mse_loss(logits1, logits2)
        gradients1 = tape.gradient(difl_loss, difl_model.trainable_weights)
        optimizer.apply_gradients(zip(gradients1, difl_model.trainable_weights))
        
        # Classification network training step
        with tf.GradientTape(persistent=True) as tape:
            temp = difl_model(xsourcebatch, training=True)
            logits3 = classification_model(temp, training=True)
            classification_loss = binary_loss(ysourcebatch, logits3)
        gradients2 = tape.gradient(classification_loss, difl_model.trainable_weights)
        gradients3 = tape.gradient(classification_loss, classification_model.trainable_weights)
        optimizer.apply_gradients(zip(gradients2, difl_model.trainable_weights))
        optimizer.apply_gradients(zip(gradients3, classification_model.trainable_weights))
        del tape
        
        train_accuracy.update_state(ysourcebatch, logits3)
        print(f"The training accuracy at batch {i+1}/{len(source_train)} is: {float(train_accuracy.result())}.")

for xbatch, ybatch in source_train:
    encodings = difl_model(xbatch, training=False)
    logits = classification_model(encodings, training=False)
    test_accuracy.update_state(ybatch, logits)
print(f"The accuracy on the training set of MNIST is: {float(test_accuracy.result())}.")
test_accuracy.reset_states()

for xbatch, ybatch in source_test:
    encodings = difl_model(xbatch, training=False)
    logits = classification_model(encodings, training=False)
    test_accuracy.update_state(ybatch, logits)
print(f"The accuracy on the test set of MNIST is: {float(test_accuracy.result())}.")
test_accuracy.reset_states()

for xbatch, ybatch in target_train:
    encodings = difl_model(xbatch, training=False)
    logits = classification_model(encodings, training=False)
    test_accuracy.update_state(ybatch, logits)
print(f"The accuracy on the training set of Inverted MNIST is: {float(test_accuracy.result())}.")
test_accuracy.reset_states()

for xbatch, ybatch in target_test:
    encodings = difl_model(xbatch, training=False)
    logits = classification_model(encodings, training=False)
    test_accuracy.update_state(ybatch, logits)
print(f"The accuracy on the test set of Inverted MNIST is: {float(test_accuracy.result())}.")
test_accuracy.reset_states()




'''
# start the training iteration
for epoch in range(epochs):
    print(f"Starting epoch {epoch+1}/{epochs} of DIFL learning!")
    
    # custom training loop for each batch of the training dataset
    for xbatch, ybatch in source_train:
        with tf.GradientTape() as tape:
            logits = classification_model(xbatch, training=True)
            train_loss = categorical_loss(ybatch, logits)
        gradients = tape.gradient(train_loss, classification_model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, classification_model.trainable_weights))
        train_accuracy.update_state(ybatch, logits)
    print(f"The current training accuracy is: {float(train_accuracy.result())}.")

    # custom loop to evaluate the corresponding test set at the end of each epoch
    for xbatch, ybatch in source_test:
        logits = classification_model(xbatch, training=False)
        test_accuracy.update_state(ybatch, logits)

    print(f"The accuracy on the validation set for epoch {epoch+1}/{epochs} is: {float(test_accuracy.result())}.")

    # reset the accuracy metrics at the end of the epoch
    train_accuracy.reset_states()
    test_accuracy.reset_states()

for xbatch, ybatch in target_train:
    logits = classification_model(xbatch, training=False)
    test_accuracy.update_state(ybatch, logits)
print(f"The accuracy on the training set of inverted MNIST is: {float(test_accuracy.result())}.")

for xbatch, ybatch in target_test:
    logits = classification_model(xbatch, training=False)
    test_accuracy.update_state(ybatch, logits)
print(f"The accuracy on the test set of inverted MNIST is: {float(test_accuracy.result())}.")
'''

        
print("\nDone!")
