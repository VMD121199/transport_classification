from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


def predict():
    labels = ["bicycle", "bike", "car"]
    test_dir = "./data/test"
    test_image_generator = ImageDataGenerator(rescale=1./255)
    test_data_gen = test_image_generator.flow_from_directory(test_dir, target_size=(
        224, 224), class_mode='categorical', batch_size=128, shuffle=False)

    loaded_model = tf.keras.models.load_model("transport_classifier.keras")
    array_probabilities = loaded_model.predict(test_data_gen)
    # proba in array_probabilities will be divide to 3 values
    # for example: [0.1 0.1 0.8] means the photo has an 80% probability of being a car.
    # we need to find the maximum of i with the index then we can find the class of the image
    probabilities = [{proba.argmax(): proba.max()}
                     for proba in array_probabilities]
    # Get the images from the generator
    images = test_data_gen[0][0]

    for img, probability in zip(images[:10], probabilities[:10]):
        # Extract the key (predicted_label) and the value (proba) from the dictionary
        predicted_label, proba = list(probability.items())[0]
        print(f'Predicted Label: {labels[predicted_label]}')
        print(f'Test accuracy: {proba:.4f}')
        plotImages(img, proba, labels[predicted_label])


def plotImages(img, probability=False, label=None):
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(img)
    ax.axis('off')
    if probability is not None and label is not None:
        ax.set_title("%.2f" % (probability*100) + "% " + label)
    plt.show()


if __name__ == "__main__":
    predict()
