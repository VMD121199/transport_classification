from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config as cfg
from data_preprocessing import random_file_test


def predict():
    labels = ["bicycle", "bike", "car"]
    test_dir = "./data/test"
    random_file_test.random_test()
    test_image_generator = ImageDataGenerator(rescale=1.0 / 255)
    test_data_gen = test_image_generator.flow_from_directory(
        test_dir,
        target_size=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH),
        class_mode="categorical",
        batch_size=128,
        shuffle=False,
    )

    loaded_model = tf.keras.models.load_model(
        "./models/output/transport_classifier.keras"
    )
    array_probabilities = loaded_model.predict(test_data_gen)
    # proba in array_probabilities will be divide to 3 values
    # for example: [0.1 0.1 0.8] means
    # the photo has an 80% probability of being a car.
    probabilities = [
        {proba.argmax(): proba.max()} for proba in array_probabilities
    ]
    # Get the images from the generator
    images = test_data_gen[0][0]

    for img, probability in zip(images[:20], probabilities[:20]):
        predicted_label, proba = list(probability.items())[0]
        print(f"Predicted Label: {labels[predicted_label]}")
        print(f"Test accuracy: {proba:.4f}")
        plotImages(img, proba, labels[predicted_label])


def plotImages(img, probability=False, label=None):
    _, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(img)
    ax.axis("off")
    if probability is not None and label is not None:
        ax.set_title("%.2f" % (probability * 100) + "% " + label)
    plt.show()


if __name__ == "__main__":
    predict()
