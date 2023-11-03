from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config as cfg


def load_generate_data(dir: str, isShuffle: bool = False):
    image_generator = ImageDataGenerator(rescale=1.0 / 255)
    return image_generator.flow_from_directory(
        dir,
        target_size=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH),
        class_mode="categorical",
        batch_size=cfg.batch_size,
        shuffle=isShuffle,
    )
