import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def prepare_data(data_dir, img_size=(150, 150), batch_size=32, validation_split=0.2):
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist.")

    classes = os.listdir(data_dir)
    if not classes:
        raise ValueError(f"No class directories found in {data_dir}.")

    for class_dir in classes:
        class_path = os.path.join(data_dir, class_dir)
        if not os.path.isdir(class_path):
            raise ValueError(f"{class_path} is not a directory.")
        images = os.listdir(class_path)
        if not images:
            raise ValueError(f"Class directory {class_path} is empty.")
        print(f"Found {len(images)} images in {class_path}.")

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    print(f"Found {train_generator.samples} images belonging to {len(train_generator.class_indices)} classes for training.")
    print(f"Found {validation_generator.samples} images belonging to {len(validation_generator.class_indices)} classes for validation.")

    return train_generator, validation_generator
