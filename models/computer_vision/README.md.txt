A neural network attempts to predict whether given images are dragons, eagles or cats using transfer learning.

## How to Use It

Put around a hundred photos of each class you want the network to be able to distinguish in separate folders in data/train/
Make sure the numbers of classes in constants.py is the same as the number of folders.
Put the photos you want to classify in data/test/test/, their classes will be printed by the console.

## How It Works

I used a dataset of a few hundreds of photos of images of cats, eagles and dragons found online, through image queries on Google Images and Bing Images. In total, 639 images are used for training, and 69 are used for validation.

The base model used is the built in MobileNetV2, of which the fully connected layers were stripped of, and replaced with two dense layers that are trained using the little dataset at hands.

Given the little amount of data, I attempt to reduce overfitting by using l2 regularization on the first dense layer. I don't attempt to fine tune the base model, it's weights are fixed.

The base model uses dropout, so we expect the training accuracy to be worse than the validation accuracy if no overfitting is seen. Given the small amount of computation my computer can do, I only tried a handful of values.

## Results

I achieve a validation accuracy of 0.91 (with training accuracy of 0.92).

Results for various hyperparameters values I tested, after 2 epochs (on the whole dataset)\
(alpha is the learning rate, lambda is the regularization strength)\
using alpha = 0.001,   lambda = 0.001, training accuracy = 0.75, val accuracy = 0.28\
using alpha = 0.0001,  lambda = 0.01,  training accuracy = 0.95, val accuracy = 0.75\
using alpha = 0.00003, lambda = 0.05,  training accuracy = 0.93, val accuracy = 0.78\
using alpha = 0.00003, lambda = 0.2,   training accuracy = 0.92, val accuracy = 0.91\
using alpha = 0.00001, lambda = 0.1,   training accuracy = 0.64, val accuracy = 0.78\
using alpha = 0.00003, lambda = 0.4,   training accuracy = 0.93, val accuracy = 0.81

I've also tried longer training times, but they just cause more overfitting.