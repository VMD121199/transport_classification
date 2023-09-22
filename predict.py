import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def predict():
    test_dir = "./data/test"
    test_image_generator = ImageDataGenerator(rescale=1./255)
    test_data_gen = test_image_generator.flow_from_directory(test_dir, target_size=(
        150, 150), class_mode='categorical', batch_size=128, shuffle=False)

    loaded_model = tf.keras.models.load_model("transport_classifier.h5")
    score = loaded_model.predict(test_data_gen)
    print(score)
    # print(f'Test loss: {score[0]:.4f}')
    # print(f'Test accuracy: {score[1]:.4f}')

if __name__ == "__main__":
    predict()
