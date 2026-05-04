import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('animal_classifier.keras')

# Class labels (Italian folder names → English names)
LABELS = {
    'cane': 'Dog 🐕',
    'cavallo': 'Horse 🐴',
    'elefante': 'Elephant 🐘',
    'farfalla': 'Butterfly 🦋',
    'gallina': 'Chicken 🐔',
    'gatto': 'Cat 🐈',
    'mucca': 'Cow 🐄',
    'pecora': 'Sheep 🐑',
    'ragno': 'Spider 🕷️',
    'scoiattolo': 'Squirrel 🐿️',
}

# Must match the order Keras assigned during training
CLASS_NAMES = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina',
               'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']

def predict(image):
    if image is None:
        return {}
    img = Image.fromarray(image).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)[0]
    return {LABELS[CLASS_NAMES[i]]: float(predictions[i]) for i in range(10)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload an animal photo"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="🐾 Animal Classifier",
    description="Upload a photo of an animal and the AI will identify it! Trained on 26,000 images with 94.6% accuracy.",
    examples=[],
    theme=gr.themes.Soft(),
)

demo.launch()
