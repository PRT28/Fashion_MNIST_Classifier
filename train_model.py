import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data() #downloaded the fashion mnist dataset from Keras

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1) #reshaped the download dataset to 60000 images with each of 28X28 size with 1 channel
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1) ##reshaped the download dataset to 10000 images with each of 28X28 size with 1 channel

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)), 
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
        ])

#optimizer used is adam as optimizes the loss with great measure
#As we are dealing with multiple classes so sparse_categorical_crossentropy is used
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy']) 

#trained the model for 40 epochs
model.fit(
    train_images,train_labels,
    epochs=40,
    verbose=1
    )
#evaluated the models accuracy on test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print("Test accuracy: ", test_acc)
model.save("model.h5")
