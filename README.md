# DIFL-Tuberculosis

This repository contains the code relating to the research of utilizing domain adaptation techniques for the purposes of Tuberculosis detection from Chest X-rays. The code mainly utilizes the Tensorflow library for both pre-processing the data for the workflow pipeline, as well as creating, training and evaluating the machine learning models. The repository is divided into 2 subsets, a **numbers** subset, where the datasets are related to optical character recognition of numbers, and a **tuberculosis** subset, where the datasets are related to tuberculosis prediction using chest x-ray images.

The numbers datasets were mainly used for evaluating correctness of algorithms, and determining the best hyper-parameters for the task of domain adaptation in tuberculosis detection.

## Numbers Domain Adaptation

This repository contains code that is related to the task of domain adaptation in optical character recognition using numbers. There are 4 main datasets that were utilized, namely:

1. MNIST Dataset
2. Inverted MNIST Dataset
3. USPS Dataset
4. SVHN Dataset

The repository contains code that was used for pre-processing the datasets, as well as the actual code for domain adaptation experimentation and evaluation.

## Tuberculosis Domain Adaptation

This repository contains code that is related to the task of detecting tuberculusis from chest x-ray images. There are 4 main datasets that were utilized, namely:

1. US Dataset from Montgomery County, Maryland, USA
2. Chinese Dataset from Shenzhen, China
3. Indian Dataset from New Delhi, India
4. TBX11K Dataset

The repository contains code that was used for pre-processing the datasets, as well as the actual code for domain adaptation experimentation and evaluation.
