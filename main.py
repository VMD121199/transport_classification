from model import model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

# This function will use for unpack data, build model then evaluation model performance


def main():
    train_dir = "./data/train"
    val_dir = "./data/validation"
    test_dir = "./data/test"

    train_image_generator = ImageDataGenerator(rescale=1./255)
    validation_image_generator = ImageDataGenerator(rescale=1./255)
    test_image_generator = ImageDataGenerator(rescale=1./255)

    train_data_gen = train_image_generator.flow_from_directory(train_dir, target_size=(
        IMG_HEIGHT, IMG_WIDTH), class_mode='categorical', batch_size=batch_size, shuffle=True)
    val_data_gen = validation_image_generator.flow_from_directory(val_dir, target_size=(
        IMG_HEIGHT, IMG_WIDTH), class_mode='categorical', batch_size=batch_size, shuffle=True)
    test_data_gen = test_image_generator.flow_from_directory(test_dir, target_size=(
        IMG_HEIGHT, IMG_WIDTH), class_mode='categorical', batch_size=batch_size, shuffle=False)

    loaded_model = model.CnnModel(input_shape=(
        IMG_HEIGHT, IMG_WIDTH, 3), num_of_class=3)
    loaded_model.model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    loaded_model.model.summary()

    history = loaded_model.model.fit(x=train_data_gen, steps_per_epoch=8,
                                     epochs=epochs, validation_data=val_data_gen, validation_steps=8, verbose=1)

    loaded_model.model.save("transport_classifier.keras")
    score = loaded_model.model.predict(test_data_gen)
    print(f'Test loss: {score[0]:.4f}')
    print(f'Test accuracy: {score[1]:.4f}')

if __name__ == "__main__":
    main()
