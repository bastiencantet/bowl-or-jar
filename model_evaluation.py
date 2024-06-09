import tensorflow as tf
import numpy as np
from keras.src.utils import load_img, img_to_array

from data_preparation import prepare_data

def evaluate_model(data_dir):
    model = tf.keras.models.load_model('model.h5')
    _, validation_generator = prepare_data(data_dir)
    loss, accuracy = model.evaluate(validation_generator)
    print(f'Validation accuracy: {accuracy}')

def predict_image(model_path, img_path):
    model = tf.keras.models.load_model(model_path)

    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        print("C'est une jarre")
    else:
        print("C'est un bol")

if __name__ == '__main__':
    data_dir = 'data'
    evaluate_model(data_dir)

    img_path = 'test.png'
    predict_image('model.h5', img_path)
    print("Test 2")
    predict_image('model.h5', 'test2.jpeg')
