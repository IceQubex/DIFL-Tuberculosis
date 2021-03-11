import os
import sys
import time
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Normalization, Rescaling, CenterCrop
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, roc_auc_score, auc

# model hyperparameters
img_height, img_width = 512,512
secondary_img_height, secondary_img_width = 63,63 
num_of_filters = 512
batch_size = 6
classification_lr = 1e-4
generator_lr = discriminator_lr = 1e-3
frequency = 25

# function to refresh a dataset
def refresh_domain_dataset():
    print("\nRefreshing the domain dataset!")
    dataset = combined_dataset.shuffle(500, reshuffle_each_iteration=False)
    dataset = dataset.batch(batch_size)
    return dataset

def refresh_classification_dataset():
    print("\nRefresing the classification dataset!")
    dataset = domain1_train_dataset.shuffle(500, reshuffle_each_iteration=False)
    dataset = dataset.batch(batch_size)
    return dataset

# function to append domain labels to a dataset
def append_domain_label(dataset, label):
    print("Enter!")
    dataset = dataset.unbatch()
    x, y = zip(*dataset)
    print("Unzipped!")
    nums = len(y)
    x = tf.reshape(x, [nums, img_height, img_width, 3])
    #y = tf.cast(y, dtype=tf.float32)
    d = [label]*nums
    d = tf.convert_to_tensor(d, dtype=tf.float32)
    x = tf.data.Dataset.from_tensor_slices((x))
    #y = tf.data.Dataset.from_tensor_slices(y)
    d = tf.data.Dataset.from_tensor_slices(d)
    print("Returning!")
    return tf.data.Dataset.zip((x, d))

# helper function to print nice strings
def prettify_number(number):
    if number == 1:
        return "1st"
    elif number == 2:
        return "2nd"
    elif number == 3:
        return "3rd"
    else:
        return f"{number}th"

# define the training step
#@tf.function(experimental_compile=True)
@tf.function
def training_step(classification_x_batch, classification_y_batch, domain_x_batch, domain_y_batch, domain_l):
    
    # classification training step
    with tf.GradientTape(persistent=True) as tape:
        logits_classification = classification_model(classification_x_batch, training=True)
        classification_batch_loss = binary_loss(classification_y_batch, logits_classification)
    classification_gradients = tape.gradient(classification_batch_loss, classification_model.trainable_weights)
    
    # gradient state updates
    classification_layer1_weights_gradients.update_state(tf.abs(tape.gradient(classification_batch_loss, classification_model.layers[1].layers[2].layers[2].trainable_weights[0])))
    classification_layer1_bias_gradients.update_state(tf.abs(tape.gradient(classification_batch_loss, classification_model.layers[1].layers[2].layers[2].trainable_weights[1])))
    classification_final_layer_weights_gradients.update_state(tf.abs(tape.gradient(classification_batch_loss, classification_model.layers[2].layers[-2].trainable_weights[0])))
    classification_final_layer_bias_gradients.update_state(tf.abs(tape.gradient(classification_batch_loss, classification_model.layers[2].layers[-2].trainable_weights[1])))
    
    del tape
    
    # update classification accuracy and loss
    classification_accuracy.update_state(classification_y_batch, logits_classification)
    classification_loss.update_state(classification_batch_loss)
    
    # update the generator and classifier models
    classification_optimizer.apply_gradients(zip(classification_gradients, classification_model.trainable_weights))

    # define the generator labels used for calculating the generator loss
    generator_labels = tf.fill([domain_l,1],0.5) 
    #generator_labels = 1-ybatchdomain
    #print(generator_labels)

    # GAN training step
    with tf.GradientTape(persistent=True) as tape:
        encodings_generator = generator_model(domain_x_batch, training=True)
        logits_discriminator = discriminator_model(encodings_generator, training=True)
        discriminator_batch_loss = binary_loss(domain_y_batch, logits_discriminator)
        generator_batch_loss = binary_loss(generator_labels, logits_discriminator)
    discriminator_gradients = tape.gradient(discriminator_batch_loss, discriminator_model.trainable_weights)
    generator_gradients = tape.gradient(generator_batch_loss, generator_model.trainable_weights)
    
    # gradient state updates
    domain_layer1_weights_gradients.update_state(tf.abs(tape.gradient(generator_batch_loss, generator_model.layers[2].layers[2].trainable_weights[0])))
    domain_layer1_bias_gradients.update_state(tf.abs(tape.gradient(generator_batch_loss, generator_model.layers[2].layers[2].trainable_weights[1])))
    domain_final_layer_weights_gradients.update_state(tf.abs(tape.gradient(discriminator_batch_loss, discriminator_model.layers[1].layers[-2].trainable_weights[0])))
    domain_final_layer_bias_gradients.update_state(tf.abs(tape.gradient(discriminator_batch_loss, discriminator_model.layers[1].layers[-2].trainable_weights[1])))
    
    del tape
     
    # update the domain accuracy and other metrics
    domain_accuracy.update_state(domain_y_batch, logits_discriminator)
    discriminator_loss.update_state(discriminator_batch_loss)
    generator_loss.update_state(generator_batch_loss)
    
    # update the generator and discriminator models
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator_model.trainable_weights))
    generator_optimizer.apply_gradients(zip(generator_gradients, generator_model.trainable_weights))

# define the testing step
#@tf.function(experimental_compile=True)
@tf.function
def testing_step(x_batch, y_batch):
    final_logits = classification_model(x_batch, training=False)
    classification_accuracy.update_state(y_batch, final_logits)

# create the log directory for this run
r = 1
while True:
    if os.path.isdir(f"runs/{r}_run"):
        r += 1
    else:
        print(f"Logs are saved as the {prettify_number(r)} run!")
        break

os.makedirs(f"runs/{r}_run")
with open(f"runs/{r}_run/hyperparameters.txt","w") as f:
    f.write(f"With Classification Training - GPU 1\n")
    f.write(f"Source Domain: {sys.argv[1]}, Target Domain: {sys.argv[2]}\n")
    f.write(f"Image Parameters: {img_height}, {img_width}\n")
    f.write(f"Secondary Image Parameters: {secondary_img_height}, {secondary_img_width}\n")
    f.write(f"Number of Filters: {num_of_filters}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Classification Learning Rate: {classification_lr}\n")
    f.write(f"Generator & Discriminator Learning Rate: {generator_lr}\n")
    f.write(f"Frequency of output to Tensorboard: {frequency}\n")

# parse input arguments
domain1_directory = sys.argv[1]
domain2_directory = sys.argv[2]
epochs = int(sys.argv[3])

print(f"\nImporting the necessary datasets..")

# import the first dataset
domain1_train_dataset = keras.preprocessing.image_dataset_from_directory(domain1_directory, labels='inferred', label_mode='binary', color_mode='rgb', batch_size=batch_size, image_size=(img_height,img_width), validation_split=0.2, subset='training', seed=99999)
domain1_test_dataset = keras.preprocessing.image_dataset_from_directory(domain1_directory, labels='inferred', label_mode='binary', color_mode='rgb', batch_size=batch_size, image_size=(img_height,img_width), validation_split=0.2, subset='validation', seed=99999)

# import the second dataset
domain2_train_dataset = keras.preprocessing.image_dataset_from_directory(domain2_directory, labels='inferred', label_mode='binary', color_mode='rgb', batch_size=batch_size, image_size=(img_height,img_width), validation_split=0.2, subset='training', seed=1)
domain2_test_dataset = keras.preprocessing.image_dataset_from_directory(domain2_directory, labels='inferred', label_mode='binary', color_mode='rgb', batch_size=batch_size, image_size=(img_height,img_width), validation_split=0.2, subset='validation', seed=1)

# calculate the length of the classification dataset
length = len(domain1_train_dataset)

print("Datasets imported!")

# adding domain labels
print("Appending domain labels to Domain1 train..")
d1_train = append_domain_label(domain1_train_dataset, 0)
print("Appending domain labels to Domain2 train..")
d2_train = append_domain_label(domain2_train_dataset, 1)

# create the classification dataset
domain1_train_dataset = domain1_train_dataset.unbatch()
domain1_train_dataset = domain1_train_dataset.cache()

# create the domain dataset
combined_dataset = d1_train.concatenate(d2_train)
combined_dataset = combined_dataset.cache()

print("Done pre-processing datasets!")


'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LIST OF MODELS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

'''
Preparing the difl generator model
'''

# define the base resnet
res = keras.applications.ResNet50(include_top=False,input_shape=(img_height,img_width,3),weights=None)

# define the entire network architecture
inputs = keras.Input(shape=(img_height,img_width,3))
x = Rescaling(scale=1.0/255)(inputs)
x = res(x)
x = layers.Conv2DTranspose(1024, (3,3), strides=(2,2), activation = layers.LeakyReLU(alpha=0.2))(x)
x = layers.Conv2DTranspose(num_of_filters, (3,3), strides=(2,2), activation = layers.LeakyReLU(alpha=0.2))(x)
x = layers.Conv2D(num_of_filters, (3,3), activation= layers.LeakyReLU(alpha=0.2))(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2DTranspose(num_of_filters, (3,3), strides=(2,2), activation = layers.LeakyReLU(alpha=0.2))(x)
x = layers.Conv2D(num_of_filters, (3,3), activation= layers.LeakyReLU(alpha=0.2))(x)
x = layers.MaxPooling2D()(x)
outputs = layers.Conv2DTranspose(num_of_filters, (3,3), strides=(2,2), activation = layers.LeakyReLU(alpha=0.2))(x)

# define the final model
generator_model = keras.Model(inputs=inputs, outputs=outputs, name="DIFL_Generator_Model")

# display the difl model summary
print(generator_model.summary())

'''
Preparing the DIFL discriminator model
'''

# define the base vgg model
vgg1 = keras.applications.VGG19(include_top=False,input_shape=(secondary_img_height,secondary_img_width,num_of_filters),weights=None)

# define the entire network architecture
inputs1 = keras.Input(shape=(secondary_img_height,secondary_img_width,num_of_filters))
x = vgg1(inputs1)
x = layers.Flatten()(x)
x = layers.Dense(256, activation = layers.LeakyReLU(alpha=0.2))(x)
x = layers.Dense(128, activation = layers.LeakyReLU(alpha=0.2))(x)
x = layers.Dense(64, activation = layers.LeakyReLU(alpha=0.2))(x)
outputs1 = layers.Dense(1, activation = 'sigmoid')(x)

# define the final model
discriminator_model = keras.Model(inputs=inputs1, outputs=outputs1, name="Discriminator_Model")

# display the model summary
print(discriminator_model.summary())


'''
Prepare the overall classifier
'''

# define the base vgg network
vgg2 = keras.applications.VGG19(include_top=False,input_shape=(secondary_img_height,secondary_img_width,num_of_filters),weights=None)

# define the entire network architecture
inputs3 = keras.Input(shape = (img_height,img_width,3))
x = generator_model(inputs3)
x = vgg2(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation = layers.LeakyReLU(alpha=0.2))(x)
x = layers.Dense(128, activation = layers.LeakyReLU(alpha=0.2))(x)
x = layers.Dense(64, activation = layers.LeakyReLU(alpha=0.2))(x)
outputs3 = layers.Dense(1, activation = 'sigmoid')(x)

# define the final model
classification_model = keras.Model(inputs=inputs3, outputs=outputs3, name="Classification_Model")

# display the model summary
print(classification_model.summary())


'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END OF LIST ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''


'''
Specify the other parameters for the networks
'''
batch = 0
epoch = 0

# instantiate the optimizer for each network
generator_optimizer = keras.optimizers.SGD(learning_rate=generator_lr, momentum=0.9)
discriminator_optimizer = keras.optimizers.SGD(learning_rate=discriminator_lr, momentum=0.9)
classification_optimizer = keras.optimizers.SGD(learning_rate=classification_lr, momentum=0.9)

# instantiate the loss function
binary_loss = keras.losses.BinaryCrossentropy()

# instantiate the accuracy metrics
domain_accuracy = keras.metrics.BinaryAccuracy()
classification_accuracy = keras.metrics.BinaryAccuracy()

# instantiate the loss metrics
discriminator_loss = keras.metrics.Mean()
generator_loss = keras.metrics.Mean()
classification_loss = keras.metrics.Mean()

# instantiate the gradient metrics
classification_layer1_weights_gradients = keras.metrics.MeanTensor()
classification_layer1_bias_gradients = keras.metrics.MeanTensor()
classification_final_layer_weights_gradients = keras.metrics.MeanTensor()
classification_final_layer_bias_gradients = keras.metrics.MeanTensor()
domain_layer1_weights_gradients = keras.metrics.MeanTensor()
domain_layer1_bias_gradients = keras.metrics.MeanTensor()
domain_final_layer_weights_gradients = keras.metrics.MeanTensor()
domain_final_layer_bias_gradients = keras.metrics.MeanTensor()

# define iterators for getting the batches
classification_iterator = iter(refresh_classification_dataset())
domain_iterator = iter(refresh_domain_dataset())

'''
Tensorboard initialization
'''
epoch_writer = tf.summary.create_file_writer(f"runs/{r}_run/epoch-logs/")
batch_writer = tf.summary.create_file_writer(f"runs/{r}_run/batch-logs/")
accuracy_writer = tf.summary.create_file_writer(f"runs/{r}_run/accuracy-logs/")

'''
Start training the model as well as its evaluation
'''
# main training loop
while True:
    
    # epochs of classification training
    while epoch < epochs:
        print(f"Starting epoch {epoch+1}/{epochs} of Domain Discrimination training!")
        
        # custom training loop for each batch in the training dataset
        for i in range(length):
            
            if i == length-1:
                print(f"Training batch {i+1}...  ")
            elif i%3 == 0:
                print(f"Training batch {i+1}...  ", end='\r')
            elif i%3 == 1:
                print(f"Training batch {i+1}.    ", end='\r')
            else:
                print(f"Training batch {i+1}..   ", end='\r')
            
            # get the batches for the classification training step
            try:
                xbatchclass, ybatchclass = classification_iterator.get_next()
            except tf.errors.OutOfRangeError:
                classification_iterator = iter(refresh_classification_dataset())
                xbatchclass, ybatchclass = classification_iterator.get_next()
            
            # get the batches for the domain (GAN) training step
            try:
                xbatchdomain, ybatchdomain = domain_iterator.get_next()
            except tf.errors.OutOfRangeError:
                domain_iterator = iter(refresh_domain_dataset())
                xbatchdomain, ybatchdomain = domain_iterator.get_next()
            
            # calculate the batch size of the domain set
            domain_length = tf.constant(len(ybatchdomain))
            
            # call the training step
            if batch == 0:
                tf.summary.trace_on(graph=True, profiler=True)
            training_step(xbatchclass, ybatchclass, xbatchdomain, ybatchdomain, domain_length)
            if batch == 0:
                with epoch_writer.as_default():
                    tf.summary.trace_export(name="Graph", step=0, profiler_outdir=f"runs/{r}_run/epoch-logs/")
            batch += 1
        
        print(f"The classification accuracy is {float(classification_accuracy.result())}.")
        print(f"The domain accuracy is {float(domain_accuracy.result())}.")

        # write to tensorboard
        with epoch_writer.as_default():
            
            # write the scalar metrics
            tf.summary.scalar("Classification Accuracy", classification_accuracy.result(), step=epoch)
            tf.summary.scalar("Classification Loss", classification_loss.result(), step=epoch)
            tf.summary.scalar("Domain Accuracy", domain_accuracy.result(), step=epoch)
            tf.summary.scalar("Generator Loss", generator_loss.result(), step=epoch)
            tf.summary.scalar("Discriminator Loss", discriminator_loss.result(), step=epoch)
                        
            # write the gradient state histogram for classification
            tf.summary.histogram("Classification Layer 1 Weight Gradients", classification_layer1_weights_gradients.result(), step=epoch)
            tf.summary.histogram("Classification Layer 1 Bias Gradients", classification_layer1_bias_gradients.result(), step=epoch)
            tf.summary.histogram("Classification Final Layer Weight Gradients", classification_final_layer_weights_gradients.result(), step=epoch)
            tf.summary.histogram("Classification Final Layer Bias Gradients", classification_final_layer_bias_gradients.result(), step=epoch)

            # write the gradient state histogram for domain
            tf.summary.histogram("Generator Layer 1 Weight Gradients", domain_layer1_weights_gradients.result(), step=epoch)
            tf.summary.histogram("Generator Layer 1 Bias Gradients", domain_layer1_bias_gradients.result(), step=epoch)
            tf.summary.histogram("Discriminator Final Layer Weight Gradients", domain_final_layer_weights_gradients.result(), step=epoch)
            tf.summary.histogram("Discriminator Final Layer Bias Gradients", domain_final_layer_bias_gradients.result(), step=epoch)

            # write the state of the networks
            tf.summary.histogram("Generator Layer 1 Weights", classification_model.layers[1].layers[2].layers[2].trainable_weights[0], step=epoch)
            tf.summary.histogram("Generator Layer 1 Bias", classification_model.layers[1].layers[2].layers[2].trainable_weights[1], step=epoch)
            tf.summary.histogram("Classifier Final Layer Weights", classification_model.layers[2].layers[-2].trainable_weights[0], step=epoch)
            tf.summary.histogram("Classifier Final Layer Bias", classification_model.layers[2].layers[-2].trainable_weights[1], step=epoch)
            tf.summary.histogram("Discriminator Final Layer Weights", discriminator_model.layers[1].layers[-2].trainable_weights[0], step=epoch)
            tf.summary.histogram("Discriminator Final Layer Bias", discriminator_model.layers[1].layers[-2].trainable_weights[1], step=epoch)
        
        epoch += 1
        
        # reset the scalar metrics
        classification_accuracy.reset_states()
        classification_loss.reset_states()
        domain_accuracy.reset_states()
        generator_loss.reset_states()
        discriminator_loss.reset_states()

        # reset the classifcation gradient metrics
        classification_layer1_weights_gradients.reset_states()
        classification_layer1_bias_gradients.reset_states()
        classification_final_layer_weights_gradients.reset_states()
        classification_final_layer_bias_gradients.reset_states()

        # reset the domain gradient metrics
        domain_layer1_weights_gradients.reset_states()
        domain_layer1_bias_gradients.reset_states()
        domain_final_layer_weights_gradients.reset_states()
        domain_final_layer_bias_gradients.reset_states()
    
        # reset the classification metric
        classification_accuracy.reset_states()
        
        # conduct testing at regular intervals
        if epoch%(frequency)==0:
            
            # test the model on 1st domain train
            for xbatch, ybatch in domain1_train_dataset.batch(batch_size):
                testing_step(xbatch, ybatch)
            domain1_train_accuracy = float(classification_accuracy.result())
            classification_accuracy.reset_states()

            # test the model on 1st domain test
            for xbatch, ybatch in domain1_test_dataset:
                testing_step(xbatch, ybatch)
            domain1_test_accuracy = float(classification_accuracy.result())
            classification_accuracy.reset_states()
            
            # test the model on 2nd domain train
            for xbatch, ybatch in domain2_train_dataset:
                testing_step(xbatch, ybatch)
            domain2_train_accuracy = float(classification_accuracy.result())
            classification_accuracy.reset_states()
            
            # test the model on 2nd domain test
            for xbatch, ybatch in domain2_test_dataset:
                testing_step(xbatch, ybatch)
            domain2_test_accuracy = float(classification_accuracy.result())
            classification_accuracy.reset_states()
            
            # print the result
            print(f"The accuracy on the 1st domain training set is: {domain1_train_accuracy}.")
            print(f"The accuracy on the 1st domain testing set is: {domain1_test_accuracy}.")
            print(f"The accuracy on the 2nd domain training set is: {domain2_train_accuracy}.")
            print(f"The accuracy on the 2nd domain testing set is: {domain2_test_accuracy}.")
            
            # write results to tensorboard
            with accuracy_writer.as_default():
                tf.summary.scalar("1st Domain Train Accuracy", domain1_train_accuracy, step=epoch)
                tf.summary.scalar("1st Domain Test Accuracy", domain1_test_accuracy, step=epoch)
                tf.summary.scalar("2nd Domain Train Accuracy", domain2_train_accuracy, step=epoch)
                tf.summary.scalar("2nd Domain Test Accuracy", domain2_test_accuracy, step=epoch)
    
    # accept user input for extra epochs
    option = "1"
    while option != "Y" and option != "y" and option != "N" and option != "n":
        option = input("Do you want to continue for additional epochs?")
    if option == "Y" or option == "y":
        epochs += int(input("Enter the number of additional epochs: "))
    if option == "N" or option == "n":
        break

print("\nDone!")
