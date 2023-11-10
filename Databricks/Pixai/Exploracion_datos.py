# Databricks notebook source
# MAGIC %md
# MAGIC # Contexto

# COMMAND ----------

# MAGIC %md
# MAGIC Desde la primera entrega, hemos profundizado en el estudio de la base de datos seleccionada, logrando identificar que las condiciones actuales de brillo e iluminación de las imágenes no son ideales para su etiquetado
# MAGIC y el posterior entrenamiento de un modelo de clasificación. Por lo tanto es imprescindible realizar un preprocesamiento adecuado para cumplir el objetivo de la primera fase el cual es "utilizar la información existente sobre la gota de tinta, asociando detalles como color, posición y frecuencia, para establecer alertas que optimicen el proceso productivo"

# COMMAND ----------

# MAGIC %md
# MAGIC # Librerías

# COMMAND ----------

# MAGIC %md
# MAGIC Para empezar se importan las librerías necesarias para hacer las primeras operaciones sobre la data cruda con el fin de convertirla en un dataset a partir del cual usar modelos de Machine Learning. 
# MAGIC
# MAGIC **Observación:** Para evitar conflictos entre las versiones de algunas librerías, se instalan previamente `numpy==1.23.5` y `opencv-python==4.8.1.78` sobre el clúster de Databircks.

# COMMAND ----------

import numpy as np
print(np.__version__)

# COMMAND ----------

import cv2
print(cv2.__version__)

# COMMAND ----------

import csv
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import minmax_scale
from pyspark.sql import SparkSession

# COMMAND ----------

# MAGIC %md
# MAGIC # Ingesta de datos desde Azure Data Lake Storage

# COMMAND ----------

# MAGIC %md
# MAGIC A continuación, realizamos la importación de datos desde un contenedor en Azure Data Lake Storage. Es importante destacar que dentro del conjunto de muestras se encuentran un total de **535** imágenes que representan varios tipos de baldosas y manchas. Previo a este proceso, se efectuó una verificación para asegurarnos de que no hubiera imágenes inutilizables o con problemas que impidieran su procesamiento en las etapas posteriores.

# COMMAND ----------

# Databricks File System
dbfs = "/dbfs"

# Ruta del contenedor en Azure Data Lake Storage que contiene las imágenes
image_container = "/mnt/raw/"
# Ruta del contenedor en Azure Data Lake Storage que contendrá las imágenes procesadas
image_processed_container = "/mnt/bronze/"

# Lista que contiene los nombres de los archivos en el contenedor
files = [file.name for file in dbutils.fs.ls(image_container)]

# Cantidad de imágenes en el contenedor
len(files)

# COMMAND ----------

# MAGIC %md
# MAGIC # Ajuste de color

# COMMAND ----------

# MAGIC %md
# MAGIC Una de las operaciones que debemos hacer es mejorar las condiciones de iluminación, contraste y color de las imágenes, por lo que exploramos varios tipos de procesamiento y ajuste con el fin de generar un dataset con imágenes cuyas condiciones sean óptimas.

# COMMAND ----------

# Elegir aleatoriamente diez imágenes para hacer varias transformaciones
random_images = random.sample(files, 10)

# Definir el valor del parámetro gamma
gamma = 1.5

# Definir el valor del parámetro de contraste
factor_contraste = 3

# Definir el valor del parámetro de saturación
factor_saturacion = 1.5

# Iterar sobre las imágenes seleccionadas aleatoriamente
for image_name in random_images:
    # Cargar la imagen original
    image = cv2.imread(f"{dbfs}{image_container}{image_name}", cv2.IMREAD_COLOR)

    # Aplicar la corrección gamma
    image_gamma = np.power(image / 255.0, 1.0 / gamma) * 255.0
    image_gamma = image_gamma.astype(np.uint8)

    # Ajustar el contraste
    image_contrast = cv2.convertScaleAbs(image, alpha=factor_contraste, beta=0)

    # Contraste + Corrección gamma
    image_contrast_gamma = np.power(image_contrast / 255.0, 1.0 / gamma) * 255.0
    image_contrast_gamma = image_contrast_gamma.astype(np.uint8)

    # Corrección gamma y contraste
    image_gamma_contrast = cv2.convertScaleAbs(image_gamma, alpha=factor_contraste, beta=0)
    image_gamma_contrast = image_gamma_contrast.astype(np.uint8)

    # Aumentar la saturación
    hsv = cv2.cvtColor(image_gamma_contrast, cv2.COLOR_BGR2HSV)
    saturated = hsv.copy()
    saturated[:, :, 1] = saturated[:, :, 1] * factor_saturacion
    # Convertir la imagen de vuelta a espacio de color BGR
    image_saturated = cv2.cvtColor(saturated, cv2.COLOR_HSV2BGR)

    # Muestra la imagen original y sus transformaciones
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 6, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")

    plt.subplot(1, 6, 2)
    plt.imshow(cv2.cvtColor(image_gamma, cv2.COLOR_BGR2RGB))
    plt.title(f"Gamma {gamma} Corregida")

    plt.subplot(1, 6, 3)
    plt.imshow(cv2.cvtColor(image_contrast, cv2.COLOR_BGR2RGB))
    plt.title(f"Contraste Ajustado")

    plt.subplot(1, 6, 4)
    plt.imshow(cv2.cvtColor(image_contrast_gamma, cv2.COLOR_BGR2RGB))
    plt.title(f"Contraste Ajustado y\nGamma {gamma} Corregida")

    plt.subplot(1, 6, 5)
    plt.imshow(cv2.cvtColor(image_gamma_contrast, cv2.COLOR_BGR2RGB))
    plt.title(f"Gamma {gamma} Corregida y\nContraste Ajustado")

    plt.subplot(1, 6, 6)
    plt.imshow(cv2.cvtColor(image_saturated, cv2.COLOR_BGR2RGB))
    plt.title("Saturación Aumentada")

    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Aplicar la mejor transformación a las imágenes de la muestra seleccionada

# COMMAND ----------

# MAGIC %md
# MAGIC Luego de comparar entre distintas transformaciones, observamos que una combinación entre correción Gamma de 1.5 y ajuste de contraste de 3 factores es lo más óptimo ya que brinda mayor iluminación y realza los colores, dando así la oportunidad de identificar características de los defectos con mayor facilidad. Por esto, procedemos a aplicar esta trasnformación a todas las imágenes de muestra.

# COMMAND ----------

# Lista para almacenar las imágenes corregidas con contraste
images = []

# Iterar sobre las imágenes de muestra
for image_name in files:
    # Cargar la imagen original
    image = cv2.imread(f"{dbfs}{image_container}{image_name}", cv2.IMREAD_COLOR)

    # Aplicar la corrección gamma
    image_gamma = np.power(image / 255.0, 1.0 / gamma) * 255.0
    image_gamma = image_gamma.astype(np.uint8)

    # Corrección gamma y contraste
    image_gamma_contrast = cv2.convertScaleAbs(image_gamma, alpha=factor_contraste, beta=0)
    image_gamma_contrast = image_gamma_contrast.astype(np.uint8)

    # Agregar la imagen corregida con contraste a la lista
    images.append(image_gamma_contrast)

# COMMAND ----------

# MAGIC %md
# MAGIC # Normalización de pixeles

# COMMAND ----------

# MAGIC %md
# MAGIC El siguiente paso consiste en normalizar los pixeles de las imagenes de manera que la representación de colores como valores queden entre 0 y 1, esto ayuda a que el procesamiento sea menos costoso. Sin embargo, debido a la dimensión y características de las imágenes, esta transformación sigue siendo exigente a nivel computacional por lo que decidimos aplicar solo a 250 imágenes aleatorias.
# MAGIC
# MAGIC **Observación:** Uno de nuestros próximos pasos es lograr paralelizar la operación de normalización de imágenes sobre el clúster de Databricks, de manera que podamos contar con una muestra más amplia de imágenes para los pasos posteriores.

# COMMAND ----------

# Elegir aleatoriamente imágenes para normalizar
random_images = random.sample(files, 250)

# COMMAND ----------

# Lista para almacenar las imágenes normalizadas
normalized_images = []

# Iterar sobre las imágenes de muestra
for image_name in random_images:
    # Cargar la imagen original
    image = cv2.imread(f"{dbfs}{image_container}{image_name}", cv2.IMREAD_COLOR)

    # Aplicar normalización
    normalized_image = image / 255.0

    # Agregar la imagen normalizada a la lista
    normalized_images.append(normalized_image)

    # Escribir la imagen normalizada como un archivo CSV
    with open(f"{dbfs}{image_processed_container}{image_name}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        for row in normalized_image:
            writer.writerow(row)

# COMMAND ----------

# MAGIC %md
# MAGIC A continuación validamos que en efecto se hayan escrito 250 archivos csv en el contenedor de destino. Cada uno de estos archivos estará asociado a una imágen normalizada, y conformará el dataset sobre el cuál implementaremos modelos de Machine Learning.

# COMMAND ----------

dest = [file.name for file in dbutils.fs.ls(image_processed_container)]
len(dest)

# COMMAND ----------

# MAGIC %md
# MAGIC # Imágenes como representación numérica

# COMMAND ----------

# MAGIC %md
# MAGIC Generamos un arreglo que almacena las imágenes transformadas, con el propósito de luego convertirlas en una representación numérica apropiada que permita llevar a cabo operaciones avanzadas de procesamiento de imágenes.

# COMMAND ----------

# Convierte la lista normalized_images en un arreglo de NumPy
X = np.array(normalized_images)
X.shape

# COMMAND ----------

# MAGIC %md
# MAGIC Como podemos observar, X es un arreglo de la forma (**250**, 800, 1120, 3), esto significa que se tiene un conjunto de datos con **250** imágenes, donde cada imagen tiene una resolución de 800x1120 píxeles, y además, cada imagen tiene tres canales de color (rojo, verde y azul o RGB) por cada píxel.

# COMMAND ----------

# MAGIC %md
# MAGIC A continuación, dejamos un ejemplo de cómo se visualiza la primera imagen en su representación númerica. Este sería el formato de los archivos en el dataset final de este entrega

# COMMAND ----------

# Visualiar numericamente la primer imagen
display(X[0])