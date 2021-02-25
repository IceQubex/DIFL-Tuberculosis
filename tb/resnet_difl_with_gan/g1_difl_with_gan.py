import os
import sys
import time
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Normalization, Rescaling, CenterCrop
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, roc_auc_score, auc

img_height, img_width = 512,512
secondary_img_height, secondary_img_width = 63,63 
num_of_filters = 512
batch_size = 6 

# function to refresh a dataset
def refresh_dataset(dataset):
    print("Refreshing!")
    dataset = dataset.unbatch()
    dataset = dataset.shuffle(50, reshuffle_each_iteration=False)
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

# parse input arguments into lists
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

length = len(domain1_train_dataset)

domain1_train_dataset = domain1_train_dataset.shuffle(1, reshuffle_each_iteration=False)
domain1_test_dataset = domain1_test_dataset.shuffle(100, reshuffle_each_iteration=False)
domain2_train_dataset = domain2_train_dataset.shuffle(100, reshuffle_each_iteration=False)
domain2_test_dataset = domain2_test_dataset.shuffle(100, reshuffle_each_iteration=False)


print("Datasets imported!")
print("Making necessary changes to datasets..")

# adding domain labels
print("Appending domain labels to Domain1 train..")
d1_train = append_domain_label(domain1_train_dataset, 0)
print("Appending domain labels to Domain1 test..")
d1_test = append_domain_label(domain1_test_dataset, 0)
print("Appending domain labels to Domain2 train..")
d2_train = append_domain_label(domain2_train_dataset, 1)
print("Appending domain labels to Domain2 test..")
d2_test = append_domain_label(domain2_test_dataset, 1)

# create the discriminator dateset
combined_dataset = d1_train.concatenate(d2_train)
combined_dataset = combined_dataset.shuffle(100)
combined_dataset = combined_dataset.shuffle(100)
combined_dataset = combined_dataset.batch(batch_size)
combined_dataset = combined_dataset.shuffle(100, reshuffle_each_iteration=False)

print("Done manipulating datasets!")


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
#x = layers.Conv2DTranspose(512, (3,3), strides=(2,2), activation = layers.LeakyReLU(alpha=0.2))(x)
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
# define input layer
inputs = keras.Input(shape=(img_height,img_width,3))

# conduct rescaling of features
x = Rescaling(scale=1.0/255)(inputs)

# apply convolution and pooling layers
x = layers.Conv2D(64, (5,5), activation = layers.LeakyReLU(alpha=0.2))(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, (3,3), activation = layers.LeakyReLU(alpha=0.2))(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(256, (3,3), activation = layers.LeakyReLU(alpha=0.2))(x)
x = layers.MaxPooling2D()(x)
#x = layers.Conv2D(512, (3,3), activation = layers.LeakyReLU(alpha=0.2))(x)
#x = layers.MaxPooling2D()(x)
#x = layers.Conv2D(256, (3,3), activation = layers.LeakyReLU(alpha=0.2))(x)
#x = layers.MaxPooling2D()(x)
outputs = layers.Conv2D(num_of_filters, (3,3), activation = layers.LeakyReLU(alpha=0.2))(x)

# define the final model
generator_model = keras.Model(inputs=inputs, outputs=outputs, name="DIFL_Generator_Model")

# display the difl model summary
print(generator_model.summary())
'''



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
# define input layer
inputs1 = keras.Input(shape=(secondary_img_height,secondary_img_width,num_of_filters))

# apply convolution and pooling layers
x = layers.Conv2D(512, (3,3), activation = layers.LeakyReLU(alpha=0.2))(inputs1)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(256, (3,3), activation = layers.LeakyReLU(alpha=0.2))(x)
x = layers.MaxPooling2D()(x)
#x = layers.Conv2D(256, (3,3), activation = layers.LeakyReLU(alpha=0.2))(x)
#x = layers.MaxPooling2D()(x)
#x = layers.Conv2D(256, (3,3), activation = layers.LeakyReLU(alpha=0.2))(x)
#x = layers.MaxPooling2D()(x)
x = layers.Conv2D(256, (3,3), activation = layers.LeakyReLU(alpha=0.2))(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation = layers.LeakyReLU(alpha=0.2))(x)
x = layers.Dense(128, activation = layers.LeakyReLU(alpha=0.2))(x)
x = layers.Dense(64, activation = layers.LeakyReLU(alpha=0.2))(x)
outputs1 = layers.Dense(1, activation = 'sigmoid')(x)

# define the final model
discriminator_model = keras.Model(inputs=inputs1, outputs=outputs1, name="DIFL_Discriminator_Model")

# display the discriminator model summary
print(discriminator_model.summary())
'''



'''
Preparing the classification model
'''

'''
# define input layer
inputs2 = keras.Input(shape = (secondary_img_height,secondary_img_width,num_of_filters))

# apply convolution and pooling layers
x = layers.Conv2D(512, (3,3), activation = layers.LeakyReLU(alpha=0.2))(inputs2)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(256, (3,3), activation = layers.LeakyReLU(alpha=0.2))(x)
x = layers.MaxPooling2D()(x)
#x = layers.Conv2D(256, (3,3), activation = layers.LeakyReLU(alpha=0.2))(x)
#x = layers.MaxPooling2D()(x)
#x = layers.Conv2D(256, (3,3), activation = layers.LeakyReLU(alpha=0.2))(x)
#x = layers.MaxPooling2D()(x)
x = layers.Conv2D(256, (3,3), activation = layers.LeakyReLU(alpha=0.2))(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation = layers.LeakyReLU(alpha=0.2))(x)
x = layers.Dense(128, activation = layers.LeakyReLU(alpha=0.2))(x)
x = layers.Dense(64, activation = layers.LeakyReLU(alpha=0.2))(x)

# define output layer
outputs2 = layers.Dense(1, activation = 'sigmoid')(x)

# define the final model
classification_model = keras.Model(inputs=inputs2, outputs=outputs2, name = "Classification_Model")
'''

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

# instantiate the optimizer for each network
generator_optimizer = keras.optimizers.SGD(lr=1e-5, momentum=0.9)
discriminator_optimizer = keras.optimizers.SGD(lr=1e-5, momentum=0.9)
classification_optimizer = keras.optimizers.SGD(lr=1e-5, momentum=0.9)

# instantiate the loss function
binary_loss = keras.losses.BinaryCrossentropy()

# instantiate the accuracy metrics
domain_accuracy = keras.metrics.BinaryAccuracy()
classification_accuracy = keras.metrics.BinaryAccuracy()

'''
Start training the model as well as its evaluation
'''
classification_iterator = iter(domain1_train_dataset)
domain_iterator = iter(combined_dataset)

while True:

    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs} of Domain Discrimination training!")
        
        # custom training loop for each batch in the training dataset
        for i in range(length):
            
            if i%3 == 0:
                print(f"Training batch {i+1}...  ", end='\r')
            elif i%3 == 1:
                print(f"Training batch {i+1}.    ", end='\r')
            else:
                print(f"Training batch {i+1}..   ", end='\r')

            # get the batches for the classification training step
            try:
                xbatchclass, ybatchclass = classification_iterator.get_next()
            except tf.errors.OutOfRangeError:
                classification_iterator = iter(refresh_dataset(domain1_train_dataset))
                xbatchclass, ybatchclass = classification_iterator.get_next()
                print(f"The classification accuracy is {float(classification_accuracy.result())}.")
                classification_accuracy.reset_states()
            
            # classification training step
            with tf.GradientTape(persistent=True) as tape:
                logits = classification_model(xbatchclass, training=True)
                classification_loss = binary_loss(ybatchclass, logits)
            classification_gradients = tape.gradient(classification_loss, classification_model.trainable_weights)
            del tape
            
            # update classification accuracy
            classification_accuracy.update_state(ybatchclass, logits)
            
            # update the generator and classifier models
            classification_optimizer.apply_gradients(zip(classification_gradients, classification_model.trainable_weights))

            #print(f"The classification accuracy on batch {i+1} is {float(classification_accuracy.result())}.")

            # get the batches for the domain (GAN) training step
            try:
                xbatchdomain, ybatchdomain = domain_iterator.get_next()
            except tf.errors.OutOfRangeError:
                domain_iterator = iter(refresh_dataset(combined_dataset))
                xbatchdomain, ybatchdomain = domain_iterator.get_next()
                print(f"The domain accuracy is {float(domain_accuracy.result())}.")
                domain_accuracy.reset_states()
            
            # define the generator labels used for calculating the generator loss
            generator_labels = tf.fill([len(xbatchdomain),1],0.5) 
            #generator_labels = 1-ybatchdomain
            #print(generator_labels)
            #time.sleep(1)

            # GAN training step
            with tf.GradientTape(persistent=True) as tape:
                encodings = generator_model(xbatchdomain, training=True)
                logits = discriminator_model(encodings, training=True)
                discriminator_loss = binary_loss(ybatchdomain, logits)
                generator_loss = binary_loss(generator_labels, logits)
            discriminator_gradients = tape.gradient(discriminator_loss, discriminator_model.trainable_weights)
            generator_gradients2 = tape.gradient(generator_loss, generator_model.trainable_weights)
            del tape
            
            # update the domain accuracy
            domain_accuracy.update_state(ybatchdomain, logits)
            
            # update the generator and discriminator models
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator_model.trainable_weights))
            generator_optimizer.apply_gradients(zip(generator_gradients2, generator_model.trainable_weights))
            
            #print(f"The domain accuracy on batch {i+1} is {float(domain_accuracy.result())}.")
            

        # reset the metric
        #classification_accuracy.reset_states()
        #domain_accuracy.reset_states()
    
    # reset the classification metric
    classification_accuracy.reset_states()

    # test the model on 1st domain train
    for xbatch, ybatch in domain1_train_dataset:
        logits = classification_model(xbatch, training=False)
        classification_accuracy.update_state(ybatch, logits)
    print(f"The accuracy on the 1st domain training set is: {float(classification_accuracy.result())}.")
    classification_accuracy.reset_states()

    # test the model on 1st domain test
    for xbatch, ybatch in domain1_test_dataset:
        logits = classification_model(xbatch, training=False)
        classification_accuracy.update_state(ybatch, logits)
    print(f"The accuracy on the 1st domain testing set is: {float(classification_accuracy.result())}.")
    classification_accuracy.reset_states()

    # test the model on 2nd domain train
    for xbatch, ybatch in domain2_train_dataset:
        logits = classification_model(xbatch, training=False)
        classification_accuracy.update_state(ybatch, logits)
    print(f"The accuracy on the 2nd domain training set is: {float(classification_accuracy.result())}.")
    classification_accuracy.reset_states()

    # test the model on 2nd domain test
    for xbatch, ybatch in domain2_test_dataset:
        logits = classification_model(xbatch, training=False)
        classification_accuracy.update_state(ybatch, logits)
    print(f"The accuracy on the 2nd domain testing set is: {float(classification_accuracy.result())}.")
    classification_accuracy.reset_states()

    # accept user input for extra epochs
    option = "1"
    while option != "Y" and option != "y" and option != "N" and option != "n":
        option = input("Do you want to continue for additional epochs?")
    if option == "Y" or option == "y":
        epochs = int(input("Enter the number of additional epochs: "))
    if option == "N" or option == "n":
        break

print("\nDone!")
