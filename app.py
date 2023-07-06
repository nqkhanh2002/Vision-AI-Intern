import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, render_template, request

# Keras and TensorFlow
from tensorflow import keras

# Load the model
loaded_model = keras.models.load_model('static/finetune_best_model.h5')

def process_image(image_path, pretrained_model):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image_array = np.array(image)
    predictions = pretrained_model.predict(np.expand_dims(image_array, axis=0))
    predicted_class = np.argmax(predictions)
    confidence_score = predictions[0, predicted_class]

    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    # Display prediction information
    print("Image Path:", image_path)
    print("Predicted Class:", predicted_class)
    print("Confidence Score:", confidence_score)
    return predicted_class

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT
    image = request.files['image']
    # Get path to image
    image_path = 'static/' + image.filename
    # image.save('static/{}.jpg'.format(COUNT))   
    # image_path = 'static/{}.jpg'.format(COUNT) 
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image_array = np.array(image)
    predictions = loaded_model.predict(np.expand_dims(image_array, axis=0))
    predicted_class = np.argmax(predictions)
    confidence_score = predictions[0, predicted_class]
    # create pred list to store prediction
    preds = []
    preds.append(predicted_class)
    preds.append(confidence_score)
    # path to save prediction
    preds.append(image_path)

    COUNT += 1

    return render_template('prediction.html', data=preds)


# @app.route('/load_img')
# def load_img():
#     global COUNT
#     return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)
