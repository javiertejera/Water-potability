#----------------------------------IMPORTAR LIBRERÍAS--------------------------------------------------------
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from PIL import Image
import wget

import os
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#to make the plotly graphs
import plotly.graph_objs as go
import chart_studio.plotly as py
from plotly.offline import iplot, init_notebook_mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

# Machine learning
from pycaret.classification import *

#----------------------------------CONFIGURACIÓN DE PÁGINA--------------------------------------------------------

st.set_page_config(page_title='Potability', layout="centered", page_icon='💧')


#----------------------------------PREPROCESAMIENTO DE DATOS--------------------------------------------------------

# Lectura de archivos

df = pd.read_csv('drinking_water_potability.csv')

# Sustituyo los valores nulos en las columnas pH, sulfatos y trihalometanos.

df['ph']=df['ph'].fillna(df.groupby(['Potability'])['ph'].transform('mean'))
df['Sulfate']=df['Sulfate'].fillna(df.groupby(['Potability'])['Sulfate'].transform('mean'))
df['Trihalomethanes']=df['Trihalomethanes'].fillna(df.groupby(['Potability'])['Trihalomethanes'].transform('mean'))

# Filtrado valores de pH según límites para el agua potable (6.5-9.5).

df.loc[df['ph'] > 9.5 , 'Potability'] = 0
df.loc[df['ph'] < 6.5 , 'Potability'] = 0

# Filtrado valores de trihalometanos según límite para el agua potable (80ug/L).

df.loc[df['Trihalomethanes'] > 80 , 'Potability'] = 0

# Filtrado valores de turbidez según límite para el agua potable (5 NTU).

df.loc[df['Turbidity'] > 5 , 'Potability'] = 0

#----------------------------------MODELO---------------------------------------------------------------------------

exp = setup(df, target='Potability', session_id=3492)
model = create_model('gbc')

#----------------------------------COMIENZA LA APP------------------------------------------------------------------

st.title('Water potability predictor') 
st.markdown("<img src='https://s3.abcstatics.com/media/ciencia/2020/10/14/juyasiudya-kv4H--1248x698@abc.jpg' width='700' height='350'>", unsafe_allow_html=True)
st.markdown('Javier Tejera')
st.markdown('15/06/23')

#Menú horizontal
menu = option_menu(

    menu_title=None,
    options=['Introducción', 'Análisis', 'Modelo', 'Predictor'],
    icons= ['droplet-half', 'clipboard-data','diagram-3', 'search'],
    default_index=0,
    orientation="horizontal",
    
)

# Pestaña introducción
if menu == 'Introducción':
    st.markdown("<h0 style='text-align:justify;'>El agua es un recurso indispensable para la vida en nuestro planeta. Pero, no solo es esencial para la vida, sino también para el desarrollo y bienestar de las sociedades, ya que, los seres humanos le damos multitud de aplicaciones que mejoran nuestro bienestar, como pueden ser la generación de energía o el regadío de cultivos. </h0>", unsafe_allow_html=True)
    st.markdown("<img src='https://www.unexport.es/wp-content/uploads/2018/06/uso-sostenible-del-agua-en-la-agricultura.jpg' width='700' height='350'>", unsafe_allow_html=True)
    st.markdown("<h0 style='text-align:justify;'> </h0>", unsafe_allow_html=True)
    st.markdown("<h0 style='text-align:justify;'>A pesar de que el agua cubre aproximadamente el 71% de la superficie de la Tierra, solo alrededor del 2.5% de ese volumen es agua dulce, y de esa cantidad, solo una pequeña fracción es agua potable. Además, el problema de la contaminación reduce año tras año la cantidad de agua potable. </h0>", unsafe_allow_html=True)
    st.markdown("<img src='https://cdn.businessinsider.es/sites/navi.axelspringer.es/public/media/image/2022/11/beber-agua-grifo-2882507.jpg?tf=1200x' width='700' height='350'>", unsafe_allow_html=True)
    st.markdown("<h0 style='text-align:justify;'> </h0>", unsafe_allow_html=True)
    st.markdown("<h0 style='text-align:justify;'>La escasez de agua potable es un desafío global que afecta a muchas regiones del mundo. Según datos de las Naciones Unidas, aproximadamente 2.200 millones de personas no tienen acceso a servicios de agua potable gestionados de forma segura, lo que pone en riesgo su salud y bienestar. </h0>", unsafe_allow_html=True)
    st.markdown("<h0 style='text-align:justify;'>Con el propósito de que las personas de todo el mundo puedan disfrutar de agua potable de manera segura. El objetivo de este estudio es proporcionar un modelo que ayude a discernir entre agua potable y agua no potable.</h0>", unsafe_allow_html=True)
    st.markdown("<img src='https://www.iagua.es/sites/default/files/images/medium/saludenfermedad.jpg' width='700' height='350'>", unsafe_allow_html=True)
    st.markdown("<h0 style='text-align:justify;'> </h0>", unsafe_allow_html=True)


# Pestaña Análisis

if menu == 'Análisis':
    menu2 = option_menu(

    menu_title=None,
    options=['Distribución muestras', 'Distribución variables', 'Correlación'],
    icons= [],
    default_index=0,
    orientation="horizontal",
    
)
    if menu2 == 'Distribución muestras':
        potable = (df['Potability'] == 1).sum()/len(df)*100
        no_potable = (df['Potability'] == 0).sum()/len(df)*100
        potabilidad =[potable,no_potable]
        labels = ['Potable', 'No potable']
        colores = ['#47f1ef','#658685']
        def func(pct, allvalues):
            absolute = int(pct / 100.*np.sum(allvalues))
            return "{:.0f}%".format(pct, absolute)
        st.markdown("<h0 style='text-align:justify;'>Como se muestra en el siguiente gráfico, para el dataset estudiado tenemos un 20% de muestras de agua que son potables y un 80% que no lo son. </h0>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize =(6, 5))
        plt.pie(x=potabilidad, colors = colores , autopct = lambda pct: func(pct, potabilidad), shadow=True, explode=[0.1,0.0])
        plt.legend(labels)
        plt.title('Proporción de potabilidad de las muestras')
        st.pyplot(fig)

    
    if menu2 == 'Distribución variables':
        st.markdown("<h0 style='text-align:justify;'>Las variables presentes en el dataset para decidir si un agua es potable o no, son: pH, dureza, sólidos, cloraminas, sulfatos, conductividad, carbono orgánico, trihalometanos y turbidez.</h0>", unsafe_allow_html=True)
        st.markdown("<h0 style='text-align:justify;'>En la siguiente gráfica se muestran las distribuciones de las variables según el agua sea potable o no potable. </h0>", unsafe_allow_html=True)
       
        fig, ax = plt.subplots(3, 3, figsize=(14, 20))

        plt.subplot(3,3,1)
        sns.kdeplot(data=df,x=df['ph'], hue="Potability", hue_order=[1,0], fill=True, palette={1:'#47f1ef', 0:'#658685'})
        plt.title('pH')
        plt.xlabel('pH')
        plt.legend(labels=['No potable', 'Potable'])

        plt.subplot(3,3,2)
        sns.kdeplot(data=df,x=df['Hardness'], hue="Potability", hue_order=[1,0], fill=True, palette={1:'#47f1ef', 0:'#658685'})
        plt.title('Dureza')
        plt.xlabel('Dureza, ppm')
        plt.legend(labels=['No potable', 'Potable'])

        plt.subplot(3,3,3)
        sns.kdeplot(data=df,x=df['Solids'], hue="Potability", hue_order=[1,0], fill=True, palette={1:'#47f1ef', 0:'#658685'})
        plt.title('Sólidos')
        plt.xlabel('[Sólidos], ppm')
        plt.legend(labels=['No potable', 'Potable'])

        plt.subplot(3,3,4)
        sns.kdeplot(data=df,x=df['Chloramines'], hue="Potability", hue_order=[1,0], fill=True, palette={1:'#47f1ef', 0:'#658685'})
        plt.title('Cloraminas')
        plt.xlabel('[Cloraminas], ppm')
        plt.legend(labels=['No potable', 'Potable'])

        plt.subplot(3,3,5)
        sns.kdeplot(data=df,x=df['Sulfate'], hue="Potability", hue_order=[1,0], fill=True, palette={1:'#47f1ef', 0:'#658685'})
        plt.title('Sulfatos')
        plt.xlabel('[Sulfatos], ppm')
        plt.legend(labels=['No potable', 'Potable'])

        plt.subplot(3,3,6)
        sns.kdeplot(data=df,x=df['Conductivity'], hue="Potability", hue_order=[1,0], fill=True, palette={1:'#47f1ef', 0:'#658685'})
        plt.title('Conductividad')
        plt.xlabel('Conductividad, μS/cm')
        plt.legend(labels=['No potable', 'Potable'])

        plt.subplot(3,3,7)
        sns.kdeplot(data=df,x=df['Organic_carbon'], hue="Potability", hue_order=[1,0], fill=True, palette={1:'#47f1ef', 0:'#658685'})
        plt.title('Carbono orgánico')
        plt.xlabel('[Carbono orgánico], ppm')
        plt.legend(labels=['No potable', 'Potable'])

        plt.subplot(3,3,8)
        sns.kdeplot(data=df,x=df['Trihalomethanes'], hue="Potability", hue_order=[1,0], fill=True, palette={1:'#47f1ef', 0:'#658685'})
        plt.title('Trihalometanos')
        plt.xlabel('[Trihalometanos], ppb')
        plt.legend(labels=['No potable', 'Potable'])

        plt.subplot(3,3,9)
        sns.kdeplot(data=df,x=df['Turbidity'], hue="Potability", hue_order=[1,0],fill=True, palette={1:'#47f1ef', 0:'#658685'})
        plt.title('Turbidez')
        plt.xlabel('Turbidez, NTU')
        plt.legend(labels=['No potable', 'Potable'])
        st.pyplot(fig)

        st.markdown("<h0 style='text-align:justify;'>Se puede observar que las distribuciones son parecidas en todas las variables tanto para el agua potable como para el agua no potable. Sin embargo, podemos ver tres claras excepciones: pH, trihalometanos y turbidez. Esto es debido a que se tuvo que realizar un limpiado de los datos atendiendo a la normativa para aguas potables. En el caso del pH había en el dataset aguas con valores fuera de los límites 6.5-9.5 que son los aceptados etiquetadas como potables. Para la concentración de trihalometanos el límite establecido es de 80 ppb y en el caso de la turbidez el valor máximo admitido es de 5 NTU, habiendo muestras etiquetadas como potables fuera de esos límites. </h0>", unsafe_allow_html=True)
        st.markdown("<h0 style='text-align:justify;'>En la siguiente gráfica se muestra una comparación un poco diferente de las distribuciones de las variables. </h0>", unsafe_allow_html=True)

        image = Image.open('feature_gbc.jpg')
        st.image(image)
        fig, axes = plt.subplots(3, 3, figsize=(16, 15))

        plt.subplot(3,3,1)
        ax = sns.boxplot(x=df['Potability'], y=df['ph'], palette={1:'#47f1ef', 0:'#658685'})
        etiquetas = ['No potable', 'Potable']
        plt.xticks(range(len(etiquetas)), etiquetas)
        ax.set(xlabel=None)
        plt.title('pH')
        plt.ylabel('pH')


        plt.subplot(3,3,2)
        ax = sns.boxplot(x=df['Potability'], y=df['Hardness'], palette={1:'#47f1ef', 0:'#658685'})
        etiquetas = ['No potable', 'Potable']
        plt.xticks(range(len(etiquetas)), etiquetas)
        ax.set(xlabel=None)
        plt.title('Dureza')
        plt.ylabel('Dureza, ppm')

        plt.subplot(3,3,3)
        ax = sns.boxplot(x=df['Potability'], y=df['Solids'], palette={1:'#47f1ef', 0:'#658685'})
        etiquetas = ['No potable', 'Potable']
        plt.xticks(range(len(etiquetas)), etiquetas)
        ax.set(xlabel=None)
        plt.title('Sólidos')
        plt.ylabel('[Sólidos], ppm')

        plt.subplot(3,3,4)
        ax = sns.boxplot(x=df['Potability'], y=df['Chloramines'], palette={1:'#47f1ef', 0:'#658685'})
        etiquetas = ['No potable', 'Potable']
        plt.xticks(range(len(etiquetas)), etiquetas)
        ax.set(xlabel=None)
        plt.title('Cloraminas')
        plt.ylabel('[Cloraminas], ppm')

        plt.subplot(3,3,5)
        ax = sns.boxplot(x=df['Potability'], y=df['Sulfate'], palette={1:'#47f1ef', 0:'#658685'})
        etiquetas = ['No potable', 'Potable']
        plt.xticks(range(len(etiquetas)), etiquetas)
        ax.set(xlabel=None)
        plt.title('Sulfatos')
        plt.ylabel('[Sulfatos], ppm')

        plt.subplot(3,3,6)
        ax = sns.boxplot(x=df['Potability'], y=df['Conductivity'], palette={1:'#47f1ef', 0:'#658685'})
        etiquetas = ['No potable', 'Potable']
        plt.xticks(range(len(etiquetas)), etiquetas)
        ax.set(xlabel=None)
        plt.title('Conductividad')
        plt.ylabel('Conductividad, μS/cm')


        plt.subplot(3,3,7)
        ax = sns.boxplot(x=df['Potability'], y=df['Organic_carbon'], palette={1:'#47f1ef', 0:'#658685'})
        etiquetas = ['No potable', 'Potable']
        plt.xticks(range(len(etiquetas)), etiquetas)
        ax.set(xlabel=None)
        plt.title('Carbono orgánico')
        plt.ylabel('[Carbono orgánico], ppm')

        plt.subplot(3,3,8)
        ax = sns.boxplot(x=df['Potability'], y=df['Trihalomethanes'], palette={1:'#47f1ef', 0:'#658685'})
        etiquetas = ['No potable', 'Potable']
        plt.xticks(range(len(etiquetas)), etiquetas)
        ax.set(xlabel=None)
        plt.title('Triahalometanos')
        plt.ylabel('[Trihalometanos], ppb')

        plt.subplot(3,3,9)
        ax = sns.boxplot(x=df['Potability'], y=df['Turbidity'], palette={1:'#47f1ef', 0:'#658685'})
        etiquetas = ['No potable', 'Potable']
        plt.xticks(range(len(etiquetas)), etiquetas)
        ax.set(xlabel=None)
        plt.title('Turbidez')
        plt.ylabel('Turbidez, NTU')
        st.pyplot(fig)


        st.markdown("<h0 style='text-align:justify;'>Se observa como en general las muestras de aguas no potables presentan un abanico más amplio de valores, tanto por arriba, como por abajo. Apareciendo menor cantidad de outliers para las muestras de agua potable. De nuevo vemos como los valores de pH aparecen truncados por abajo, mientras que, los valores de trihalometanos y turbidez aparecen truncados por arriba debido al preprocesamiento de los datos.</h0>", unsafe_allow_html=True)

    if menu2 == 'Correlación':

        st.markdown("<h0 style='text-align:justify;'>En el siguiente gráfico se muestran las correlaciones entre las diferentes variables, incluyendo la variable objetivo, potabilidad. </h0>", unsafe_allow_html=True)
        mask = np.triu(df.corr())
        fig,ax = plt.subplots(figsize=(10,7))
        sns.heatmap(df.corr(),cmap='Blues',annot=True, mask=mask)
        plt.title('Correlación entre variables')
        st.pyplot(fig)

        st.markdown("<h0 style='text-align:justify;'>Se ve como las variables se correlacionan muy poco con la potabilidad, salvo el pH y los trihalometanos que parecen tener una relación significativa. </h0>", unsafe_allow_html=True)

# Pestaña modelo

if menu == 'Modelo':
    st.markdown("<h0 style='text-align:justify;'>En la siguiente tabla se muestran los resultados obtenidos del entrenamiento de diferentes modelos utilizando la librería pycaret.</h0>", unsafe_allow_html=True)
    results = pd.read_csv('results_model_potability.csv')
    st.dataframe(results)
    st.markdown("<h0 style='text-align:justify;'>Podemos observar como tenemos cuatro modelos con la misma exactitud del 90%. El siguiente factor a tener en cuenta para la elección del modelo con el que vamos a trabajar va a ser la precisión. A mayor precisión menor número de falsos positivos cometerá el modelo. En el caso del agua potable no nos interesa tener falsos positivos, ya que, un falso positivo significa etiquetar como potable un agua que no lo es. Lo que puede acarrear graves consecuencias, como enfermedades en la población e incluso muertes. Teniendo esto en cuenta, el modelo elegido fue el Gradient boosting classifier, con una precisión del 83%.</h0>", unsafe_allow_html=True)
    st.markdown("<h0 style='text-align:justify;'>De manera simplificada el modelo gradient boosting se basa en el ajuste de diferentes modelos sencillos en los que cada modelo aprende del anterior.</h0>", unsafe_allow_html=True)
    st.markdown("<h0 style='text-align:justify;'>Vamos a explicar nuestro modelo de una manera más visual.</h0>", unsafe_allow_html=True)
    st.markdown("<h0 style='text-align:justify;'>En primer lugar observamos la representación de la curva ROC. Esta representación nos da una idea de la exactitud del modelo, cuanto más área (AUC) halla bajo ella mejor predecirá nuestro modelo.</h0>", unsafe_allow_html=True)
    plot_model(model, 'auc', display_format='streamlit')
    st.markdown("<h0 style='text-align:justify;'> </h0>", unsafe_allow_html=True)
    st.markdown("<h0 style='text-align:justify;'>En esta segunda gráfica observamos la matriz de confusión del modelo. Como hemos elegido un modelo con una precisión más alta que la sensibilidad (Recall), se ve que el ratio de falsos positivos es menor que el de falsos negativos.</h0>", unsafe_allow_html=True)
    plot_model(model, 'confusion_matrix', display_format= 'streamlit')
    st.markdown("<h0 style='text-align:justify;'> </h0>", unsafe_allow_html=True)
    st.markdown("<h0 style='text-align:justify;'>A continuación se muestra la representación de la curva de aprendizaje del modelo. Se puede considerar que el modelo es bueno, ya que, la validación del modelo va aumentando según disminuye la cantidad de datos de entrenamiento.</h0>", unsafe_allow_html=True)
    image = Image.open('learning_curve.jpg')
    st.image(image)
    st.markdown("<h0 style='text-align:justify;'> </h0>", unsafe_allow_html=True)
    st.markdown("<h0 style='text-align:justify;'>Por último, se muestra una figura con la importancia que tiene cada variable a la hora de realizar las predicciones con el modelo seleccionado.</h0>", unsafe_allow_html=True) 
    image = Image.open('feature_gbc.jpg')
    st.image(image)
    st.markdown("<h0 style='text-align:justify;'> </h0>", unsafe_allow_html=True)
    
    



# Pestaña predictor

if menu == 'Predictor':
    exp = setup(df, target='Potability', session_id=3492)
    model = create_model('gbc')
    save_model(model, 'gbc')
    # Primero coloco los sliders para elegir las características
    ph = st.slider(label = 'pH', min_value =1.0, max_value = 14.0 , value = 7.0, step = 0.1)
    Hardness = st.slider(label = 'Dureza, ppm', min_value =40.0, max_value = 350.0 , value = 200.0, step = 0.1)
    Solids = st.slider(label = '[Sólidos], ppm', min_value =300.0, max_value = 61300.0 , value = 1000.0, step = 0.1)
    Chloramines = st.slider(label = '[Cloraminas], ppm', min_value = 0.0, max_value = 13.5 , value = 5.0, step = 0.10)
    Sulfate = st.slider(label = '[Sulfatos], ppm', min_value = 100.0, max_value = 500.0 , value = 250.0, step = 0.10)
    Conductivity = st.slider(label = 'Conductividad, mS/cm', min_value = 150.0, max_value = 800.0 , value = 200.0, step = 0.10)
    Organic_carbon = st.slider(label = '[Carbono orgánico], ppm', min_value = 0.0, max_value = 30.0 , value = 3.0, step = 0.10)
    Trihalomethanes = st.slider(label = '[Trihalometanos], ppb', min_value = 0.0, max_value = 125.0 , value = 15.0, step = 0.10)
    Turbidity = st.slider(label = 'Turbidez, NTU', min_value = 1.0, max_value = 7.0 , value = 3.0, step = 0.10)
    
    #Ahora preparo el dataframe con todas las características que se usarán para predecir la potabilidad
    features = {'ph':ph, 'Hardness': Hardness,'Solids':Solids, 'Chloramines':Chloramines, 'Sulfate':Sulfate, 'Conductivity':Conductivity, 'Organic_carbon':Organic_carbon, 'Trihalomethanes':Trihalomethanes, 'Turbidity':Turbidity}

    df = pd.DataFrame(features, index = [0])
    prediction = (model, df)
    dfFeatures = pd.DataFrame([features])

    if st.button('¿Potabilidad?'):    
        prediction = predict_model(model, dfFeatures)
        if prediction.iloc[0,9] == 0:
           st.write('Su agua no es potable')
        else:
           st.write('Su agua es potable')


    

    
