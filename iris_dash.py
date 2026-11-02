from PIL import Image
import streamlit as st
from joblib import load

model = load('data/iris_model.joblib')

col1, col2 = st.columns(2)

# with col1:
sepal_length = col1.number_input('Sepal length', step=0.1, value=4.0, min_value=0.0, max_value=15.0)
sepal_width = col1.number_input('Sepal width', step=0.1, value=3.0, min_value=0.0, max_value=15.0)
  
# with col2:
petal_length = col2.number_input('Petal length', step=0.1, value=4.0, min_value=0.0, max_value=15.0)
petal_width = col2.number_input('Petal width', step=0.1, value=2.0, min_value=0.0, max_value=15.0)

c1, c2, c3 = st.columns(3)

if c2.button('Make a prediction') == True:
  
  prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
  if prediction == 0:
    # setosa
    image = Image.open('images/iris_setosa.png')
  elif prediction == 1:
    # versicolor
    image = Image.open('images/iris_versicolor.png')
  else:
    # virginica
    image = Image.open('images/iris_virginica.png')
  
  c2.write(f"Your prediction is: ")
  c2.image(image)
  
