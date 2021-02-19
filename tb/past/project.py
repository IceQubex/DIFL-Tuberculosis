import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Normalization, Rescaling, CenterCrop
from tensorflow.keras import layers

train_directory = sys.argv[1]
epochs = int(sys.argv[4])
test1_directory = sys.argv[2]
test2_directory = sys.argv[3]

# split into train and validation sets
print("Reading the training data!")
train_dataset = keras.preprocessing.image_dataset_from_directory(train_directory, labels = 'inferred', label_mode = 'binary', color_mode = 'grayscale', batch_size = 16, image_size=(200,200), validation_split = 0.2, subset = 'training', seed = 123)
validation_dataset = keras.preprocessing.image_dataset_from_directory(train_directory, labels = 'inferred', label_mode = 'binary', color_mode = 'grayscale', batch_size = 16, image_size=(200,200), validation_split = 0.2, subset = 'validation', seed = 123)

# import the datasets to test on
test1_dataset = keras.preprocessing.image_dataset_from_directory(test1_directory, labels = 'inferred', label_mode = 'binary', color_mode = 'grayscale', batch_size = 16, image_size=(200,200))
test2_dataset = keras.preprocessing.image_dataset_from_directory(test2_directory, labels = 'inferred', label_mode = 'binary', color_mode = 'grayscale', batch_size = 16, image_size=(200,200))

'''
Preparing the model
'''

# define input layer
inputs = keras.Input(shape=(200,200,1))
# conduct rescaling of features
x = Rescaling(scale=1.0/255)(inputs)
# apply convolution and pooling layers
x = layers.Conv2D(32,(3,3), activation = 'relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32,(3,3), activation = 'relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, (3,3), activation = 'relu')(x)
# flatten into dense layers
x = layers.Flatten()(x)
x = layers.Dense(64, activation = 'relu')(x)
# define output layer
outputs = layers.Dense(1, activation = 'sigmoid')(x)
# define the final model
model = keras.Model(inputs=inputs, outputs=outputs)
# display the model summary
print(model.summary())

'''
Specificy the other parameters for the model
'''

# instantiate the optimizer
optimizer = keras.optimizers.Adam(lr=0.001)

# instantiate the loss function
loss_function = keras.losses.BinaryCrossentropy()

# instantiate the accuracy metrics
train_accuracy = keras.metrics.BinaryAccuracy()
val_accuracy = keras.metrics.BinaryAccuracy()
test_accuracy = keras.metrics.BinaryAccuracy()

# instantiate the log writers for TensorBoard
train_writer = tf.summary.create_file_writer("logs/train")
val_writer = tf.summary.create_file_writer("logs/val")
test1_writer = tf.summary.create_file_writer("logs/test1")
test2_writer = tf.summary.create_file_writer("logs/test2")

'''
Start training the model as well as its evaluation
'''

# start the training iteration
for epoch in range(epochs):
    print(f"Starting epoch {epoch+1}/{epochs}!")
    
    # custom training loop for each batch of the training dataset
    for step, (train_x_batch, train_y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(train_x_batch, training=True)
            train_loss = loss_function(train_y_batch, logits)
        gradients = tape.gradient(train_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        train_accuracy.update_state(train_y_batch, logits)
        print(f"The current training accuracy for batch {step+1}/{len(train_dataset)} is: {float(train_accuracy.result())}.")
    
    # to log the results for TensorBoard for training set
    with train_writer.as_default():
        tf.summary.scalar("Loss", train_loss, step=epoch)
        tf.summary.scalar("Accuracy", train_accuracy.result(), step=epoch)

    # custom loop to evaluate the validation set at the end of each epoch
    for val_x_batch, val_y_batch in validation_dataset:
        logits = model(val_x_batch, training=False)
        val_loss = loss_function(val_y_batch, logits)
        val_accuracy.update_state(val_y_batch, logits)

    # to log the results for TensorBoard for validation set
    with val_writer.as_default():
        tf.summary.scalar("Loss", val_loss, step=epoch)
        tf.summary.scalar("Accuracy", val_accuracy.result(), step=epoch)

    print(f"The accuracy on the validation set for epoch {epoch+1}/{epochs} is: {float(val_accuracy.result())}.")
    
    # reset the accuracy metrics at the end of the epoch
    val_accuracy.reset_states()
    train_accuracy.reset_states()

    '''
    Final testing on the test sets
    '''

    # start the testing loop
    for test_x_batch, test_y_batch in test1_dataset:
        logits = model(test_x_batch, training = False)
        test_loss = loss_function(test_y_batch, logits) 
        test_accuracy.update_state(test_y_batch, logits)
    
    with test1_writer.as_default():
        tf.summary.scalar("Loss", test_loss, step=epoch)
        tf.summary.scalar("Accuracy", test_accuracy.result(), step=epoch)


    # display result of the test set
    print("\n-----TEST DATASET 1-----\n")
    print(f"The accuracy on the first test set is: {float(test_accuracy.result())}.")

    # reset the accuracy metric on the test set
    test_accuracy.reset_states()

    # start the testing loop
    for test_x_batch, test_y_batch in test2_dataset:
        logits = model(test_x_batch, training = False)
        test_loss = loss_function(test_y_batch, logits)
        test_accuracy.update_state(test_y_batch, logits)

    with test2_writer.as_default():
        tf.summary.scalar("Loss", test_loss, step=epoch)
        tf.summary.scalar("Accuracy", test_accuracy.result(), step=epoch)

    # display result of the test set
    print("\n-----TEST DATASET 2-----\n")
    print(f"The accuracy on the second test set is: {float(test_accuracy.result())}.")
    
    # reset the accuracy metric on the test set
    test_accuracy.reset_states()

print("\nDone!")



            



'''
# compile the model
model.compile(
    loss = keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
    )

# train and evaluate the model
model.fit(train_dataset, epochs=epochs,verbose =1)
model.evaluate(validation_dataset, verbose = 1)

print("\nTesting on different domain..\n")
model.evaluate(test_dataset, verbose = 1)
'''
