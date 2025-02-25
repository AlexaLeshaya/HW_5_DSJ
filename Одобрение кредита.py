import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.title('😇 La-la-la gushosh kachala')

st.write('hah, new project')

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
with st.expander('Data'):
  st.write('X')
  X_raw = df.drop('species', axis=1)
  st.dataframe(X_raw)

  st.write('y')
  y_raw = df.species
  st.dataframe(y_raw)

with st.sidebar:
  st.header("Enter features: ")
  Island = st.selectbox('Island', ('Torgersen', 'Biscoe', 'Dream'))
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 44.5)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.3)
  flipper_length_mm =  st.slider('Flipper length (mm)', 32.1, 59.6, 44.5)
  body_mass_g =  st.slider('Body mass (g)', 32.1, 59.6, 44.5)
  sex = st.selectbox('Gender', ("male", "female"))

st.subheader('Data visualization')
fig = px.scatter(
  df,
  x='bill_length_mm',
  y='bill_depth_mm',
  color='island',
  title='Bill lenghts vs Bill  by Island'
)
st.plotly_chart(fig)



fig2 = px.histogram(
  df,
  x='body_mass_g',
  nbins = 30,
  title = 'Distribution of body mass'
)
st.plotly_chart(fig2)


data = {
    'island': Island,
    'bill_length_mm': bill_length_mm,
    'bill_depth_mm': bill_depth_mm,
    'flipper_length_mm': flipper_length_mm,
    'body_mass_g': body_mass_g,
    'sex': sex
}
input_df = pd.DataFrame(data, index=[0])
input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
  st.write('**Input penguin**')
  st.dataframe(input_df)
  st.write('**Combined penguins data (input row + original dataset)**')
  st.dataframe(input_penguins)
encode = ['Island','sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)
X = df_penguins[1:]
input_row = df_penguins[:1]
target_mapper = {'Adelie':0,'Chinstrap':1,'Gentoo':2}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write('**Encoded X (Input penguin)**')
  st.dataframe(input_row)
  st.write('**Encoded y**')
  st.write(y)
base_rf = RandomForestClassifier(random_state = 42)
base_rf.fit(X,y)
prediction = base_rf.predict(input_row)
prediction_proba = base_rf.predict_proba(input_row)
df_prediction_proba = pd.DataFrame(prediction_proba, columns = ['Adelie','Chinstrap','Gentoo'])

st.subheader('Predicted Species')
st.dataframe(
  df_prediction_proba,
  column_config={
    'Adelie':st.column_config.ProgressColumn(
      'Adelie',
            format='%f',
            width='medium',
            min_value=0,
            max_value=1
        ),
        'Chinstrap': st.column_config.ProgressColumn(
            'Chinstrap',
            format='%f',
            width='medium',
            min_value=0,
            max_value=1
        ),
        'Gentoo': st.column_config.ProgressColumn(
            'Gentoo',
            format='%f',
            width='medium',
            min_value=0,
            max_value=1
        ),
    },
    hide_index=True
)

penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(f"Predicted species: **{penguins_species[prediction][0]}**")
