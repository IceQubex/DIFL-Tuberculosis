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

def append_domain_label(dataset, label):
    dataset = dataset.unbatch()
    x, y = zip(*dataset)
    nums = len(y)
    x = tf.reshape(x, [nums, 512, 512, 1])
    y = tf.cast(y, dtype=tf.float32)
    d = [label]*nums
    d = tf.convert_to_tensor(d, dtype=tf.float32)
    x = tf.data.Dataset.from_tensor_slices((x))
    y = tf.data.Dataset.from_tensor_slices(y)
    d = tf.data.Dataset.from_tensor_slices(d)
    return tf.data.Dataset.zip((x, y, d))

# parse input arguments into lists
source_directory = sys.argv[1]
target_directory = sys.argv[2]
epochs = int(sys.argv[3])

print(f"\nImporting the necessary datasets..")
# import the source dataset
source_train = keras.preprocessing.image_dataset_from_directory(source_directory, labels='inferred', label_mode='binary', color_mode='grayscale', batch_size=16, image_size=(512,512), validation_split=0.2, subset='training', seed=123)
source_test = keras.preprocessing.image_dataset_from_directory(source_directory, labels='inferred', label_mode='binary', color_mode='grayscale', batch_size=16, image_size=(512,512), validation_split=0.2, subset='validation', seed=123)

# import the target dataset
target_train = keras.preprocessing.image_dataset_from_directory(target_directory, labels='inferred', label_mode='binary', color_mode='grayscale', batch_size=16, image_size=(512,512), validation_split=0.2, subset='training', seed=132)
target_test = keras.preprocessing.image_dataset_from_directory(target_directory, labels='inferred', label_mode='binary', color_mode='grayscale', batch_size=16, image_size=(512,512), validation_split=0.2, subset='validation', seed=132)

print("Datasets imported!")

print("Making necessary changes to datasets..")

# adding domain labels
print("Appending domain labels to source train..")
source_train = append_domain_label(source_train, 0)
print("Appending domain labels to source test..")
source_test = append_domain_label(source_test, 0)
print("Appending domain labels to target train..")
target_train = append_domain_label(target_train, 1)
print("Appending domain labels to target test..")
target_test = append_domain_label(target_test, 1)

# create the discriminator dateset
domain_dataset = source_train.concatenate(target_train)
domain_dataset = domain_dataset.shuffle(100000)
domain_dataset = domain_dataset.shuffle(100000)
domain_dataset = domain_dataset.batch(16)

# convert to datasets
source_train = source_train.batch(16)
source_test = source_test.batch(16)
target_train = target_train.batch(16)
target_test = target_test.batch(16)

print("Done manipulating datasets!")

'''
Preparing the difl model
'''
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

'''
Preparing the classification model
'''
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

'''
Preparing the domain distinguisher model
'''

# define input layer
inputs2 = keras.Input(shape=(512,512,1))

# conduct rescaling of features
x = Rescaling(scale=1.0/255)(inputs2)

# apply convolution and pooling layers
x = layers.Conv2D(32, (5,5), activation = layers.LeakyReLU(alpha=0.05))(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, (3,3), activation = layers.LeakyReLU(alpha=0.05))(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, (3,3), activation = layers.LeakyReLU(alpha=0.05))(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(256, (3,3), activation = layers.LeakyReLU(alpha=0.05))(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation = layers.LeakyReLU(alpha=0.05))(x)
x = layers.Dense(64, activation = layers.LeakyReLU(alpha=0.05))(x)
outputs2 = layers.Dense(1, activation = 'sigmoid')(x)

# define the final model
discriminator_model = keras.Model(inputs=inputs2, outputs=outputs2, name="Discriminator_Model")

# display the discriminator model summary
print(discriminator_model.summary())

'''
Specify the other parameters for the model
'''

# instantiate the optimizer
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)

# instantiate the loss function
#categorical_loss = keras.losses.SparseCategoricalCrossentropy()
#mse_loss = keras.losses.MeanSquaredError()
binary_loss = keras.losses.BinaryCrossentropy()

# instantiate the accuracy metrics
#train_accuracy = keras.metrics.SparseCategoricalAccuracy()
#test_accuracy = keras.metrics.SparseCategoricalAccuracy()
domain_accuracy = keras.metrics.BinaryAccuracy()

'''
Start training the model as well as its evaluation
'''
#source_iterator = iter(source_train)
#target_iterator = iter(target_train)

for epoch in range(epochs):
    print(f"Starting epoch {epoch+1}/{epochs} of Domain Discrimination training!")

    # custom training loop for each batch in the training dataset
    for i, (xbatch, ybatch, dbatch) in enumerate(domain_dataset):
        
        # training step
        with tf.GradientTape() as tape:
            logit = discriminator_model(xbatch, training=True)
            loss = binary_loss(dbatch, logit)
        gradients = tape.gradient(loss, discriminator_model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, discriminator_model.trainable_weights))
        domain_accuracy.update_state(dbatch, logit)
        print(f"The domain accuracy on batch {i+1} is {float(domain_accuracy.result())}.")
    
    # shuffle the dataset
    domain_dataset = refresh_dataset(domain_dataset)

    # reset the metric
    domain_accuracy.reset_states()

# test the model on normal MNIST train
for xbatch, ybatch, dbatch in source_train:
    logit = discriminator_model(xbatch, training=False)
    domain_accuracy.update_state(dbatch, logit)
#normal_train = float(test_accuracy.result())
print(f"The accuracy on the training set of normal MNIST is: {float(domain_accuracy.result())}.")
domain_accuracy.reset_states()

# test the model on normal MNIST train
for xbatch, ybatch, dbatch in source_test:
    logit = discriminator_model(xbatch, training=False)
    domain_accuracy.update_state(dbatch, logit)
#normal_train = float(test_accuracy.result())
print(f"The accuracy on the testing set of normal MNIST is: {float(domain_accuracy.result())}.")
domain_accuracy.reset_states()

# test the model on normal MNIST train
for xbatch, ybatch, dbatch in target_train:
    logit = discriminator_model(xbatch, training=False)
    domain_accuracy.update_state(dbatch, logit)
#normal_train = float(test_accuracy.result())
print(f"The accuracy on the training set of inverted MNIST is: {float(domain_accuracy.result())}.")
domain_accuracy.reset_states()

# test the model on normal MNIST train
for xbatch, ybatch, dbatch in target_test:
    logit = discriminator_model(xbatch, training=False)
    domain_accuracy.update_state(dbatch, logit)
#normal_train = float(test_accuracy.result())
print(f"The accuracy on the testing set of inverted MNIST is: {float(domain_accuracy.result())}.")
domain_accuracy.reset_states()

'''
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
'''        
print("\nDone!")
#print(f"{normal_train};{normal_test};{inverted_train};{inverted_test}")
