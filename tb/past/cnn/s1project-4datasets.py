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

# function to plot the roc_curve
def plot_roc_curve(fpr,tpr, name, label):
    plt.plot(fpr,tpr, label = label)
    plt.title(name)
    plt.axis([0,1,0,1])
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.savefig(f"results/test-{name}.png")

# parse input arguments into lists
names = ["TBX", "China", "India", "US"]
datasets_directory = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]
epochs = [int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])]

with open('results/results.txt','w') as f:
    f.write("Results..")

# loop to iterate through the datasets
for d in range(len(datasets_directory)):

    with open('results/results.txt','a') as f:
        f.write(f"\n\n{names[d]}:\n\n")
    
    print(f"\nTraining the model on the {names[d]} dataset!")

    # import the training dataset
    print(f"\nImporting the necessary datasets..")
    train_dataset = keras.preprocessing.image_dataset_from_directory(datasets_directory[d], labels = 'inferred', label_mode = 'binary', color_mode = 'grayscale', batch_size = 16, image_size=(200,200), validation_split = 0.2, subset = 'training', seed = 123)

    # import the other datasets to be tested on
    test_datasets = []
    for k in range(len(datasets_directory)):
        if k == d:
            test_datasets.append(keras.preprocessing.image_dataset_from_directory(datasets_directory[k], labels='inferred', label_mode='binary', color_mode='grayscale', batch_size=16, image_size=(200,200), validation_split=0.2, subset='validation', seed=123))
            continue
        test_datasets.append(keras.preprocessing.image_dataset_from_directory(datasets_directory[k], labels='inferred', label_mode='binary', color_mode='grayscale', batch_size=16, image_size=(200,200)))
    print("\nDone importing the datasets!")

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
    test_accuracy = keras.metrics.BinaryAccuracy()
    
    '''
    Start training the model as well as its evaluation
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

    '''
    Final testing on the test sets
    '''

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
        with open("results/results.txt",'a') as f:
            f.write(f"The accuracy on the {names[k]} dataset is: {float(test_accuracy.result())}.\n")
            f.write(f"The AUC value on the {names[k]} dataset is: {roc_auc_score(y_true, y_pred)}.\n")

        # reset the accuracy metric on the test set
        test_accuracy.reset_states()
    
    # close the current plt figure and start a new plt figure
    plt.close()
           
print("\nDone!")
