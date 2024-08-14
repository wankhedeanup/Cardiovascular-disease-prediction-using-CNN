from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

app = Flask(__name__)

# Load your trained model
model = load_model('C:/Users/Lenovo/Downloads/cardiovascular_disease_model.h5')



def preprocess_image(image_path):
    # Load image with target size (224, 224)
    img = load_img(image_path, target_size=(224, 224))
    
    # Convert the image to grayscale
    img = img.convert('L')  # 'L' mode is for grayscale
    
    # Convert image to array
    img = img_to_array(img)
    
    # Reshape the array to (1, 224, 224, 1) to match model input shape
    img = img.reshape(1, 224, 224, 1)
    
    # Normalize pixel values to be between 0 and 1
    img = img.astype('float32')
    img /= 255.0
    
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # Save the file temporarily
            image_path = "temp_image.png"
            file.save(image_path)
            
            # Preprocess the image
            processed_image = preprocess_image(image_path)
            
            # Make a prediction
            prediction = model.predict(processed_image)
            
            return f'Prediction: {prediction}'
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
