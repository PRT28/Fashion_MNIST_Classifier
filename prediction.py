import tensorflow as tf
import cv2
import numpy as np

labels= ['T-shirt', 'trousers', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'] #labels for the dataset

model=tf.keras.models.load_model('model.h5')#loaded the pretrained model saved from 'train_model.py'.
path=input() #input the name of the image provided in the repo.
img=cv2.imread(path) #reading the image by opencv
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #the image is read in bgr mode by defaul so converting it to rgb format.
img=cv2.resize(img, (28,28)) #resizing the image to 28X28
img = img.reshape(1, 28, 28, 1) #reshaping the image so that it can be fed to neural network.
img = tf.cast(img, tf.float32) #casting the pixels of the image to float32
preds = model.predict(img) #pridicting the class of the image.
label=np.argmax(preds) #calculating the index of the label
print(labels[label]) #display the class
