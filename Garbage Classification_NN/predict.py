import tensorflow as tf
import numpy as np

classes = ['Battery', 'Cardboard', 'Clothes', 'Glass', 'Metal', 'Paper', 'Plastic']

cnn_model = tf.keras.models.load_model(
    "CNN_Model_Architecture/garbage_cnn_model.keras"
)

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