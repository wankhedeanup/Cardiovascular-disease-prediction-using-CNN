from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model(r'C:\Users\Lenovo\Downloads\ML_project\cardiovascular_disease_model.h5')


def preprocess_image(image):
    print(f"Original Image Mode: {image.mode}")  # Debug: Check image mode
    
    image = image.resize((128, 128))  # Resize to 128x128 pixels
    image = np.array(image)
    
    print(f"Image shape after resizing: {image.shape}")  # Debug: Check shape after resizing
    
    if image.ndim == 2:  # Convert grayscale images to RGB
        image = np.stack([image]*3, axis=-1)
        print("Converted grayscale to RGB")  # Debug: Confirm conversion
    
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    print(f"Final preprocessed image shape: {image.shape}")  # Debug: Final image shape
    
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            try:
                image = Image.open(file)
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                
                result = 'sick' if prediction[0][0] > 0.5 else 'normal'
                
                print(f"Prediction: {result}")  # Debug: Output prediction result

                return render_template('index_2.html', prediction=result)
            except Exception as e:
                return str(e)
        else:
            return "No file uploaded or file is not an image."
    return render_template('index_2.html')

if __name__ == '__main__':
    app.run(debug=True)




