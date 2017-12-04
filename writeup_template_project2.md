#**Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[train_hist]: ./examples/train_hist.png "Training dataset histogram"
[valid_hist]: ./examples/valid_hist.png "Validation dataset histogram"
[test_hist]: ./examples/test_hist.png "Testing dataset histogram"


[image2]: ./examples/color.png "Color"
[image9]: ./examples/gray.png "Gray"
[image3]: ./examples/augmented.png "Random Noise"

[image0]: ./myimages/0 "Traffic Sign 0"
[image14]: ./myimages/14 "Traffic Sign 1"
[image24]: ./myimages/24 "Traffic Sign 2"
[image35]: ./myimages/35 "Traffic Sign 3"
[image40]: ./myimages/40 "Traffic Sign 4"
[image7]: ./myimages/7 "Traffic Sign 5"
[image25]: ./myimages/25 "Traffic Sign 6"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

## Rubric Points
Here I will provide a reference to the sections below that address each individual rubric. The rubric points and descriptions for this project may be found [here](https://review.udacity.com/#!/rubrics/481/view).

- Dataset Exploration
  - [Dataset Summary](#dataset-summary)
  - [Exploratory Visualization](#exploratory-visualization)
- Design and Test a Model Architecture
  - [Preprocessing](#preprocessing)
  - [Model Architecture](#model-architecture)
  - [Model Training](#model-training)
  - [Solution Approach](#solution-approach)
- Test A Model On New Images
  - [Acquiring New Images](#acquiring-new-images)
  - [Performance on New Images](#performance-on-new-images)
  - [Model Certainty Softmax Probabilities](#model-certainty-softmax-probabilities)


## Data Set Summary & Exploration

### Dataset Summary

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of the original training set is *34799*
* The size of the validation set is *4410*
* The size of test set is *12630*
* The shape of a traffic sign image is *32*x*32*x*3*, 32x32 pixels with 3 color channels
* The number of unique classes/labels in the data set is *43*. This is the number of types of signs considered in these datasets.


### Exploratory Visualization

For each of the three data sets,  a histogram show the distribution of sign types,
the _relative_ distributions have roughly the same structure in each histogram.

![alt Training][train_hist]
![alt Validation][valid_hist]
![alt Test][test_hist]


## Design and Test a Model Architecture

### Preprocessing
1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color carries no extra information in imgage recognition.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image9]

Normalization was done to give dataset equal variance, and reduce distortion.

Normalizing was according to pixel = (pixel - 128)/128.0.

In order to add more data, augmentation was performed. Random images were chosen to apply a rotational effect,
and the rotated images were added to the dataset. A total of 2200 images per class was the result of augmentation.

The final dataset has 43 classes with 2200 images each for a total training set of 94,600 images

The following the histogram of augmented dataset, the distrition of each sign type (class) is the same.

![alt text][image3]


### Model Architecture

2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.)
Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray scale image
| Convolution           | 5x5 filter, 1x1 stride, VALID padding, outputs 28x28x6
| RELU					|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6
| Convolution 	        | 5x5 filter, 1x1 stride, VALID padding, outputs 10x10x16
| RELU					|
| Max Pooling			|2x2 stride, outputs 5x5x16
| Flatten				| Input 5x5x16 Output 400
| Layer 3 Fully Connected	| input 400 Output 200
| RELU					| 
| DROPOUT				| keep 80%
| Layer 4 Fully Connected | Input 200, Output 100
| RELU					|
| DROPOUT				| keep 80%
| Layer 5 Fully Connected | Input 100, Output 43
| OUTPUT                | 43 (# of sign types)

 


3. Describe how you trained your model. The discussion can include the type of optimizer,
the batch size, number of epochs and any hyperparameters such as learning rate.

The basic Lenet architecture was used for the model. Dropout was included on the
fully connected layers. The Adam optimizer was used for computing and applying gradients.
The following is hypyter parameter values:
EPOCHS = 10
BATCH_SIZE = 128
rate = 0.005
mu = 0
sigma = 0.1

A drop out (keep 80%) was used in the training and keep 100% in the validation.

####4. Describe the approach taken for finding a solution and getting the validation
set accuracy to be at least 0.93. Include in the discussion the results on the training,
validation and test sets and where in the code these were calculated.
Your approach may have been an iterative process, in which case, outline the steps you took to get
 to the final solution and why you chose those steps. Perhaps your solution involved an already well known
 implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 0.993
* validation set accuracy of 0.953
* test set accuracy of  0.932
* web set accuracy of 0.857

Drop out greatly improve accuracy


###Test a Model on New Images

1. Choose seven German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][image0] ![alt text][image7] ![alt text][image14]
![alt text][image24] ![alt text][image25] ![alt text][image35] ![alt text][image40]

The image that the network did not recognize correctly was speed limit 100. It mistook it as "keep right" sign. Perhaps because the two signs
are simular in gray scale

2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign
| Road narrows on the right     			| Road narrows on the right
| Ahead Only					| Ahead Only
| Roundabout	      		| Roundabout	
| 100 KM / h		|  Keep right
|  20 KM / h		|  20 KM / h
| Road work         | Road work
The model was able to correctly recognize 6 of the 7 traffic signs, which gives an accuracy of 86%.

3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax
probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign
type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric,
visualizations can also be provided such as bar charts)


| Image | Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------:|:---------------------------------------------:| 
| 25 | [1.00 0.00 0.00 0.00 0.00] | [25 29 31 30 24]   									|
| 24 | [1.00 0.00 0.00 0.00 0.00] | [24 18 29 30 11]										 |
| 35 | [1.00 0.00 0.00 0.00 0.00] | [35 34  9 36  3] 											|
| 40 | [1.00 0.00 0.00 0.00 0.00] | [40 12 21 11  1]				 				|
| 14 | [1.00 0.00 0.00 0.00 0.00] | [14  3 34 17 32]      							|
| 0 | [1.00 0.00 0.00 0.00 0.00] | [0 4 1 8 5]      							|
| 7 | [0.97 0.03 0.00 0.00 0.00] | [38 14  2  8  3]     							|


Only image 7 was INCORRECTLY classified.
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


