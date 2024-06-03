import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the dataset
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train.reshape(-1, 28, 28, 1))

# Build the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with data augmentation
model.fit(datagen.flow(x_train.reshape(-1, 28, 28, 1), y_train, batch_size=32),
          epochs=5,
          validation_data=(x_test.reshape(-1, 28, 28, 1), y_test))

# Save the model
model.save('handwritten.model')

# Load the saved model
model = tf.keras.models.load_model('handwritten.model')

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

# Predict on the test set and plot confusion matrix
y_pred = model.predict(x_test.reshape(-1, 28, 28, 1))
y_pred_classes = np.argmax(y_pred, axis=1)
plot_confusion_matrix(y_test, y_pred_classes, classes=range(10))

# Function to predict and show digit images
def predict_and_show_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not read image {image_path}")
            return
        img = cv2.resize(img, (28, 28))  # Resize image to 28x28
        img = np.invert(np.array([img]))  # Invert colors
        img = img / 255.0  # Normalize the image
        prediction = model.predict(img.reshape(-1, 28, 28, 1))
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

# Iterate over digit images in the directory
image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    predict_and_show_image(f"digits/digit{image_number}.png")
    image_number += 1

# Interactive prediction
def interactive_prediction():
    while True:
        image_path = input("Enter the path of the digit image (or 'exit' to quit): ")
        if image_path.lower() == 'exit':
            break
        if os.path.isfile(image_path):
            predict_and_show_image(image_path)
        else:
            print(f"File {image_path} does not exist. Please try again.")

interactive_prediction()

# Visualize model architecture
def visualize_model_architecture(model):
    tf.keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
    img = plt.imread('model_architecture.png')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

visualize_model_architecture(model)
