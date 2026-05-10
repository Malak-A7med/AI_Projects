import tensorflow as tf
import numpy as np
import os

classes = ['Battery', 'Cardboard', 'Clothes', 'Glass', 'Metal', 'Paper', 'Plastic']

model_path = os.path.join(
    os.path.dirname(__file__),
    "CNN_Model_Architecture/garbage_cnn_model.keras"
)

cnn_model = tf.keras.models.load_model(model_path)

mobilenet_model = tf.keras.models.load_model(
    "MOBILENETV2/mobilenet_v2_final.keras"
)

def predict_image(processed_image, model_name):

    if model_name == "CNN":
        model = cnn_model
    else:
        model = mobilenet_model

    prediction = model.predict(processed_image)

    predicted_index = np.argmax(prediction)

    confidence = np.max(prediction)

    predicted_class = classes[predicted_index]

    return predicted_class, confidence, prediction[0]
