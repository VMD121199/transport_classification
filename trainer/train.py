from models.cnn import CNN


def train_model(optimizer, input_shape: tuple, train_data, val_data):
    loaded_model = CNN(input_shape=input_shape, num_of_class=3)
    loaded_model.model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    loaded_model.model.summary()

    loaded_model.model.fit(
        x=train_data,
        steps_per_epoch=8,
        epochs=15,
        validation_data=val_data,
        validation_steps=15,
        verbose=1,
    )

    return loaded_model
