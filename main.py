import config as cfg
from trainer.train import train_model
from trainer.helpers import load_generate_data


# Bicycle: 0, Bike: 1, Car: 2
def main():
    train_dir = "./data/train"
    val_dir = "./data/validation"

    train_data_gen = load_generate_data(train_dir, True)
    val_data_gen = load_generate_data(val_dir, True)
    model = train_model(
        input_shape=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH, 3),
        train_data=train_data_gen,
        val_data=val_data_gen,
        optimizer="adam",
    ).model
    model.save("models/output/transport_classifier.keras")


if __name__ == "__main__":
    main()
