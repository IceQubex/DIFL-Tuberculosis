import os
import sys
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Normalization, Rescaling, CenterCrop
from tensorflow.keras import layers, activations 
from sklearn.metrics import roc_curve, roc_auc_score, auc

if len(sys.argv) != 9:
    print("Wrong number of arguments provided! Please use the following format:")
    print("\"python3 bottleneckfeatures-resnet.py <dataset1> <dataset2> <dataset3> <dataset4> <epochs for dataset1> <epochs for dataset2> <epochs for dataset3> <epochs for dataset4>\"")
    sys.exit()


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
names = ["China", "India", "US", "TBX"]
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
    train_dataset = keras.preprocessing.image_dataset_from_directory(datasets_directory[d], labels = 'inferred', label_mode = 'binary', color_mode = 'rgb', batch_size = 16, image_size=(200,200), validation_split = 0.2, subset = 'training', seed = 123)

    # import the other datasets to be tested on
    test_datasets = []
    for k in range(len(datasets_directory)):
        if k == d:
            test_datasets.append(keras.preprocessing.image_dataset_from_directory(datasets_directory[k], labels='inferred', label_mode='binary', color_mode='rgb', batch_size=16, image_size=(200,200), validation_split=0.2, subset='validation', seed=123))
            continue
        test_datasets.append(keras.preprocessing.image_dataset_from_directory(datasets_directory[k], labels='inferred', label_mode='binary', color_mode='rgb', batch_size=16, image_size=(200,200)))
    print("\nDone importing the datasets!")

    # getting the bottleneck features
    bottleneck_model = keras.applications.ResNet50(weights = 'imagenet', include_top = False, input_shape=(200,200,3))
    #print(bottleneck_model.summary())
    

    print("\nExtracting features using ResNet50 model..\n")
    
    # custom loop to save the extracted features 
    for i, (xbatch, ybatch) in enumerate(train_dataset):
        xbatch = xbatch/255
        if i == 0:
            features = bottleneck_model(xbatch)
            labels = ybatch
        else:
            features = tf.concat([features, bottleneck_model(xbatch)], 0)
            labels = tf.concat([labels, ybatch], 0)
        print(f"Extracted features on {i+1}/{len(train_dataset)} batches..")

    print("\nDone!")
    
    feature_dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(16)

    # prepare the top classification model
    top_inputs = keras.Input(shape=(7,7,2048))
    y = layers.Flatten()(top_inputs)
    y = layers.Dense(256, activation = 'relu')(y)
    y = layers.Dense(64, activation = 'relu')(y)
    top_outputs = layers.Dense(1, activation = 'sigmoid')(y)
    top_model = keras.Model(inputs=top_inputs, outputs=top_outputs, name = f"{names[d]}_Top_Model")
    print(top_model.summary())

    # top model parameters
    top_optimizer = keras.optimizers.SGD(lr=1e-4, momentum = 0.9)
    top_loss_function = keras.losses.BinaryCrossentropy()
    top_accuracy = keras.metrics.BinaryAccuracy()

    # start the training iteration of the top model to approximate the weights
    i = acc = 0
    while True:
        print(f"\nStarting epoch {i+1} on the {names[d]} dataset for the top model!")
        
        # custom training loop for each batch of the training dataset
        for top_i, (top_xbatch, top_ybatch) in enumerate(feature_dataset):
            with tf.GradientTape() as tape:
                top_logits = top_model(top_xbatch, training=True)
                top_loss = top_loss_function(top_ybatch, top_logits)
            top_gradients = tape.gradient(top_loss, top_model.trainable_weights)
            top_optimizer.apply_gradients(zip(top_gradients, top_model.trainable_weights))
            top_accuracy.update_state(top_ybatch, top_logits)
            print(f"The current training accuracy in batch {top_i+1}/{len(feature_dataset)} is: {float(top_accuracy.result())}.")
        
        # if condition to break out of training loop
        if abs(float(top_accuracy.result()) - acc) < 0.000000001 or i>20:
            break
        acc = float(top_accuracy.result())

        # reset accuracy and increment loop counter
        top_accuracy.reset_states()
        i+=1

    '''
    Preparing the main model
    '''

    # import the ResNet50 model
    resmodel = keras.applications.ResNet50(weights = 'imagenet', input_shape = (200,200,3), include_top = False)
    for layer in resmodel.layers[:-32]:
        layer.trainable = False

    # define input layer
    inputs = keras.Input(shape=(200,200,3))
    # conduct rescaling of features
    x = Rescaling(scale=1.0/255)(inputs)
    # utilize the resnet Model
    x = resmodel(x)
    # the output layer
    outputs = top_model(x)

    # define the final model
    main_model = keras.Model(inputs=inputs, outputs=outputs, name = f"{names[d]}_Main_Model")
    
    # display the model summary
    print(main_model.summary())
    time.sleep(5)

    '''
    Specificy the other parameters for the model
    '''

    # instantiate the optimizer
    optimizer = keras.optimizers.SGD(lr=1e-7,momentum=0.9)

    # instantiate the loss function
    loss_function = keras.losses.BinaryCrossentropy()

    # instantiate the accuracy metrics
    train_accuracy = keras.metrics.BinaryAccuracy()
    test_accuracy = keras.metrics.BinaryAccuracy()
    
    '''
    Start training the model as well as its evaluation
    '''

    # start the training iteration of the main model
    for epoch in range(epochs[d]):
        print(f"\nStarting epoch {epoch+1}/{epochs[d]} on the {names[d]} dataset!")
        
        # custom training loop for each batch of the training dataset
        for i, (xbatch, ybatch) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = main_model(xbatch, training=True)
                train_loss = loss_function(ybatch, logits)
            gradients = tape.gradient(train_loss, main_model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, main_model.trainable_weights))
            train_accuracy.update_state(ybatch, logits)
            print(f"The current training accuracy in batch {i+1}/{len(train_dataset)} is: {float(train_accuracy.result())}.")
        
        train_accuracy.reset_states()

        # custom loop to evaluate the corresponding test set at the end of each epoch
        for xbatch, ybatch in test_datasets[d]:
            logits = main_model(xbatch, training=False)
            test_accuracy.update_state(ybatch, logits)
            print(loss_function(ybatch, logits))

        print(f"\nThe accuracy on the validation set for epoch {epoch+1}/{epochs[d]} is: {float(test_accuracy.result())}.")
        ''' 
        if epoch % 10 == 0:
            user = input("Break?[Y/N]")

            if user == "Y" or user == "y":
                break

        # custom breaking conditions
        if d == 0 and float(test_accuracy.result()) > 0.815:
            break
        if d == 1 and float(test_accuracy.result()) > 0.72:
            break
        if d == 2 and float(test_accuracy.result()) > 0.9:
            break
        if d == 3 and float(test_accuracy.result()) > 0.914:
            break
        '''
        # reset the accuracy metrics at the end of the epoch
        test_accuracy.reset_states()

    '''
    Final testing on the test sets
    '''

    # start the testing loop 1
    for k in range(len(test_datasets)):

        y_pred = []
        y_true = []
        for xbatch, ybatch in test_datasets[k]:
            logits = main_model(xbatch, training = False)
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
