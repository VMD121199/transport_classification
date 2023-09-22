import os
from model import model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

### This function will use for unpack data, build model then evaluation model performance
def main():
    train_dir = "./data/train"
    val_dir = "./data/validation"
    test_dir = "./data/test"

    total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
    total_val = sum([len(files) for r, d, files in os.walk(val_dir)])
    total_test = sum([len(files) for r, d, files in os.walk(test_dir)])

    train_image_generator = ImageDataGenerator(rescale=1./255)
    validation_image_generator = ImageDataGenerator(rescale=1./255)
    test_image_generator = ImageDataGenerator(rescale=1./255)

    train_data_gen = train_image_generator.flow_from_directory(train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='categorical', batch_size=batch_size, shuffle=True)
    val_data_gen = validation_image_generator.flow_from_directory(val_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='categorical', batch_size=batch_size, shuffle=True)
    test_data_gen = test_image_generator.flow_from_directory(test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='categorical', batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    main()
