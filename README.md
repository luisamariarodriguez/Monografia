# Monografia

### Documentación de la solución

A continuación describimos brevemente la solución implementada por medio de un notebook en Databricks, en el cual se lleva a cabo el preprocesamiento de imágenes para su uso en modelos de aprendizaje automático, con el fin de construir el dataset que se requiere para la implementación del proyecto *Evolución PIXAI: Sistema de visión artificial aplicada a la inspección de baldosas*. A continuación, se documenta el código por secciones:

#### **1. Contexto:**
   - Se proporciona una introducción al contexto del proyecto, explicando que se ha identificado que las condiciones de brillo e iluminación de las imágenes no son ideales para su etiquetado y entrenamiento de un modelo de clasificación.

#### **2. Librerías:**
   - Se importan las librerías necesarias, incluyendo numpy, OpenCV, matplotlib, random, scikit-learn, y pyspark.sql.

#### **3. Ingesta de datos desde Azure Data Lake Storage:**
   - Se define la ruta del contenedor en Azure Data Lake Storage que contiene las imágenes.
   - Se muestra la cantidad de imágenes en el contenedor.
   - Se listan los archivos en el contenedor.

#### **4. Ajuste de color:**
   - Se eligen aleatoriamente diez imágenes para aplicar diversas transformaciones y ajustes de color.
   - Se realiza la corrección gamma, ajuste de contraste y aumento de saturación.
   - Se muestra visualmente cada imagen original y sus transformaciones.

#### **5. Aplicar la mejor transformación a las imágenes de la muestra seleccionada:**
   - Se decide aplicar una combinación específica de corrección gamma y ajuste de contraste a todas las imágenes de muestra.

#### **6. Normalización de pixeles:**
   - Se eligen aleatoriamente 100 imágenes para aplicar la normalización de pixeles.
   - Se normalizan las imágenes dividiendo cada valor de píxel por 255.
   - Se guarda cada imagen normalizada como un archivo CSV en un contenedor.

#### **7. Validación de la escritura de archivos CSV:**
   - Se valida la escritura de los archivos CSV en el contenedor de destino.

#### **8. Imágenes como representación numérica:**
   - Se convierte la lista de imágenes normalizadas en un arreglo de NumPy.
   - Se muestra la forma del arreglo resultante.

#### **9. Visualización de la primera imagen en su representación numérica:**
   - Se visualiza la representación numérica de la primera imagen normalizada.

### Observaciones y siguientes pasos:
- Se realiza un ajuste de color (corrección gamma, ajuste de contraste) para mejorar las condiciones de las imágenes.
- Se normalizan los píxeles de 100 imágenes y se guardan como archivos CSV.
- Se convierten las imágenes normalizadas en un arreglo de NumPy para su procesamiento posterior.

### Siguientes pasos sugeridos:
- Implementar la normalización de imágenes de manera paralela en el clúster de Databricks para manejar un conjunto más amplio de datos.
- Utilizar el conjunto de datos preprocesado para entrenar modelos de aprendizaje automático en pasos posteriores del proyecto.

### Entregables:
- Origen de datos: Si bien el origen de los datos provienen de un contenedor en Azure Data Lake Storage y este ha sido la fuente principal durante el desarrollo en Databricks, hemos decidido facilitar el acceso a estas imágenes adicionalmente almacenándolas en el directorio Monografia/Origen de datos. Esta medida se toma con el objetivo de proporcionar una referencia fácil y accesible para la consulta directa de las imágenes.
- Dataset final: Similarmente, el destino de nuestro dataset final se encuentra ubicado sobre un contenedor de Azure Data Lake Storage, sin embargo, decidimos facilitar su acceso y exploración por medio del directorio Monografia/Dataset final.
- Solución: El código que permite realizar el preprocesamiento de las imágenes se encuentra disponible en el archivo Monografia/Databricks/Pixai/Exploracion_datos.py. Además, para mayor conveniencia de los usuarios interesados, se ofrece acceso directo al workspace de Databricks donde se implementó la solución. Esto facilitará la consulta, ejecución y revisión del código en caso de ser necesario.
