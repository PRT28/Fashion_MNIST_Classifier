import tensorflow as tf
import cv2
import numpy as np

labels= ['T-shirt', 'trousers', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

model=tf.keras.models.load_model('model.h5')
path=input()
img=cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img=cv2.resize(img, (28,28))
img = img.reshape(1, 28, 28, 1)
img = tf.cast(img, tf.float32)
preds = model.predict(img)
label=np.argmax(preds)
print(labels[label])
