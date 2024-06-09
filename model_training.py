import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from data_preparation import prepare_data

def build_model(input_shape=(150, 150, 3)):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def train_model(data_dir, epochs=20, batch_size=32):
    train_generator, validation_generator = prepare_data(data_dir, batch_size=batch_size)

    if train_generator.samples == 0 or validation_generator.samples == 0:
        raise ValueError("Not enough images in the dataset to train and validate the model.")

    model = build_model()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=epochs
    )

    model.save('model.h5')

if __name__ == '__main__':
    data_dir = 'data'
    train_model(data_dir)
