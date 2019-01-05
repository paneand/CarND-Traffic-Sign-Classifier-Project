**Traffic Sign Classifier Project​**
===================================

[Self-Driving Car Engineer Nanodegree Program](http://www.udacity.com/drive)

Description
-----------

In this project, Deep neural networks and Convolutional neural networks concepts
will be used to classify traffic signs. A model will be trained and valuated so
that it can classify traffic sign images using the [German Traffic Sign
Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After
the model is trained, it will be tested also on new images of German traffic
signs that found on the web.

This project has been developed using also a [Jupyter
Notebook](https://jupyter.org/) which made the prototyping very quick.

Dependencies
------------

In order to successfully run the Jupyter Notebook containing the code, several
dependencies need to be installed.

[CarND-Term1-Starter-Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)
contains the starter kit needed for this purpose. During the development
[Anaconda platform](https://www.anaconda.com/) was used.

Implementation and model architecture design 
---------------------------------------------

The pipeline for this project, found in the Jupyter notebook
[Traffic_Sign_Classifier.ipynb](Traffic_Sign_Classifier.ipynb) has been:

1.  Load the data set

2.  Explore, summarize and visualize the data set. Validation set size training
    set size ratio is around 13%.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The histograms of the 3 datasets show that the frequency of each traffic sign is
approximately homogenous between the datasets, which ensure consistency when
validating and testing the model. In each dataset, the frequency of each traffic
sign is not constant: some of them are really frequent, others not. This results
in a an expected error distribution when predicting that is not constant between
the traffic signs

1.  Pre process the data:

    1.  Shuffle all the sets (this is needed to ensure a proper convergence to
        the minimum of the loss function)

    2.  Standardize the data set (i.e. shift and stretch the distribution so
        that the mean equals 0 and the standard deviation equals 1)

    Grayscale transformation was not applied since color information is a
    relevant feature concerning traffic signs

2.  Train and test a model architecture:

    The LeNet architecture was the starting model used for this project:

| Layer Type                                                      | Input    | Padding | Stride | Dropout | Output   |
|-----------------------------------------------------------------|----------|---------|--------|---------|----------|
| Convolution 5x5 for each image channel, 6 times (i.e. 5x5x3x6)  | 32x32x3  | VALID   | 1      | 0.9     | 28x28x6  |
| Max pooling 2x2                                                 | 28x28x6  | VALID   | 2      | /       | 14x14x6  |
| Convolution 5x5 for each channel, 16 times (i.e. 14x14x6x16)    | 14x14x6  | VALID   | 1      | 0.9     | 10x10x16 |
| Max pooling 2x2                                                 | 10x10x16 | VALID   | 2      | /       | 5x5x16   |
| Flatten                                                         | 5x5x16   | /       | /      | /       | 400      |
| Fully connected                                                 | 400      |         |        | 0.9     | 120      |
| Fully connected                                                 | 120      |         |        | 0.9     | 84       |
| Fully connected                                                 | 84       |         |        | 0.9     | 43       |

    After several experiments the following training hyperparameters were used
    to train the model:

    Epochs: 60

    Loss function optimizer: AdamOptimizer

    Batch size: 100

    Learning rate: 0.001

    No early termination mechanism was used during training to prevent
    overfitting, as dropout was implemented. Since LeNet was designed to
    recognize characters, this model worked well without re-engineering it too
    much given the task similarities.

3.  Make predictions of the new images using the model, i.e. 5 images of german
    traffic signs found on the web (stored in `./web_signs/`) were used to test
    the model against “out in the wild” images

4.  Analyze the softmax probabilities of the new images

 

Results and further improvements
--------------------------------

Results of the predictions on the sets were:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Training set prediction accuracy = 0.997
Validation set prediction accuracy = 0.943
Test set prediction accuracy = 0.935
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since the accuracy on the validation set and test set are similar, it is likely
that this model was trained in a proper manner preventing under/over fitting the
problem.

Concerning the 5 images found on the web, one contained a traffic sign that the
model was not trained to detect (“No emergency stop” traffic sign). Out of the
other 4 images, 3 were correctly detected resulting in an prediction accuracy of
75%.

Further improvements of this model could comprise early termination during
training, and several data augmentation tecniques to make the model more robust.
