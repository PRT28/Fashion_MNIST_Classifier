# Fashion_MNIST_Classifier

Fashion MNIST is a dataset containing 10 different classes of fashion articles:
- 0 T-shirt/top
- 1 Trouser
- 2 Pullover
- 3 Dress
- 4 Coat
- 5 Sandal
- 6 Shirt
- 7 Sneaker
- 8 Bag
- 9 Ankle boot

Dataset consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28×28 grayscale image

![fmnist](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F549262%2Fd6f4f6e13fa211c9e773479566d89ac9%2FExample-for-fashion-MNIST-Each-class-is-represented-by-nine-cases.png?generation=1576784453715625&alt=media)

## Brief Summary
The model contains the following layers:
- Convoltional layers (To extract feature of images)
- MaxPool layers (To reduce the size of feature map)
- Flatten layer
- Dense layers

After training the model the model has been saved and  then used saved model to predict the images.
In predictions.py the image is preprocessed according to the input_shape of the layer 0 and fed to the model.
The model gives the predictions in form of probabilty and then which is used to figure out the class.

## Usage Of The Repo

To use the repo, clone the repo and run __'predictions.py'__ and enter the name of the image(when asked)provided in the repo with extension, or you can also use your own image with specified path.
The model is already trained and saved in repo
The code used to train and savve the model is also provided in __'train_model.py'__
