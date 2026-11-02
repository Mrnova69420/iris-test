from pathlib import Path
from PIL import Image
import streamlit as st
from joblib import load

# Resolve absolute path to current file's directory
BASE_DIR = Path(__file__).parent.resolve()

# Load model
model = load(BASE_DIR / 'data' / 'iris_model.joblib')

col1, col2 = st.columns(2)

sepal_length = col1.number_input('Sepal length', step=0.1, value=4.0, min_value=0.0, max_value=15.0)
sepal_width = col1.number_input('Sepal width', step=0.1, value=3.0, min_value=0.0, max_value=15.0)
 
petal_length = col2.number_input('Petal length', step=0.1, value=4.0, min_value=0.0, max_value=15.0)
petal_width = col2.number_input('Petal width', step=0.1, value=2.0, min_value=0.0, max_value=15.0)

c1, c2, c3 = st.columns(3)

if c2.button('Make a prediction'):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    
    if prediction == 0:
        image = Image.open(BASE_DIR / 'images' / 'iris_setosa.png')
    elif prediction == 1:
        image = Image.open(BASE_DIR / 'images' / 'iris_versicolor.png')
    else:
        image = Image.open(BASE_DIR / 'images' / 'iris_virginica.png')
    
    c2.write("Your prediction is:")
    c2.image(image)
