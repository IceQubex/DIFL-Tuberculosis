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

directory = sys.argv[1]

datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range = 5)

# import the training dataset
print(f"\nImporting the necessary datasets..")
train_datagen = datagen.flow_from_directory(directory, class_mode = None, color_mode = 'grayscale', batch_size = 32, target_size=(512,512), save_to_dir = "datasets/tbx/1", save_format = "png")

#print(tf.shape(train_dataset))

#for item, label in train_dataset:
 #   print(tf.shape(item))

i = 0
for x_batch in train_datagen:
    print(f"{i} images generated!")
    i += 32
    if i > 7500:
        break





















'''
# start the training iteration
for epoch in range(epochs[d]):
    print(f"Starting epoch {epoch+1}/{epochs[d]} on the {names[d]} dataset!")
    
    # custom training loop for each batch of the training dataset
    for xbatch, ybatch in train_dataset:
        with tf.GradientTape() as tape:
            logits = model(xbatch, training=True)
            train_loss = loss_function(ybatch, logits)
        gradients = tape.gradient(train_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        train_accuracy.update_state(ybatch, logits)
    print(f"The current training accuracy is: {float(train_accuracy.result())}.")

    # custom loop to evaluate the corresponding test set at the end of each epoch
    for xbatch, ybatch in test_datasets[d]:
        logits = model(xbatch, training=False)
        test_accuracy.update_state(ybatch, logits)

    print(f"The accuracy on the validation set for epoch {epoch+1}/{epochs[d]} is: {float(test_accuracy.result())}.")
    
    if d==1 and float(test_accuracy.result()) > 0.7:
        break

    # reset the accuracy metrics at the end of the epoch
    train_accuracy.reset_states()
    test_accuracy.reset_states()


Final testing on the test sets


# start the testing loop 1
for k in range(len(test_datasets)):

    y_pred = []
    y_true = []
    for xbatch, ybatch in test_datasets[k]:
        logits = model(xbatch, training = False)
        test_accuracy.update_state(ybatch, logits)
        y_pred += list(logits)
        y_true += list(tf.reshape(ybatch, [-1]))

    # calculate the roc curve 
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plot_roc_curve(fpr,tpr,names[d], names[k])
    
    # display result of the test set
    print(f"\nThe accuracy on the {names[k]} dataset is: {float(test_accuracy.result())}.")
    print(f"The AUC value on the {names[k]} dataset is: {roc_auc_score(y_true, y_pred)}.")
    
    # write the results to a file
    with open("results.txt",'a') as f:
        f.write(f"The accuracy on the {names[k]} dataset is: {float(test_accuracy.result())}.\n")
        f.write(f"The AUC value on the {names[k]} dataset is: {roc_auc_score(y_true, y_pred)}.\n")

    # reset the accuracy metric on the test set
    test_accuracy.reset_states()

# close the current plt figure and start a new plt figure
plt.close()
'''       
print("\nDone!")
