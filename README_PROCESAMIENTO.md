# Técnicas de Procesamiento de Imágenes Implementadas

## Estructura del Proyecto

```
Proyecto-Gestos-de-la-mano-Visi-n-Artificial/
├── Preprocesamiento.ipynb              # Preprocesamiento básico original
├── Preprocesamiento_Completo.ipynb     # Notebook con todas las técnicas
├── tecnicas_procesamiento.py           # Script Python con implementaciones
└── data/
    ├── inputs/                         # Imágenes originales
    └── procesadas/                     # Resultados organizados
        ├── transformaciones/           # Transformaciones geométricas
        ├── contraste_brillo/           # Ajustes de contraste/brillo
        ├── filtros_espaciales/         # Filtros aplicados
        └── segmentacion/               # Técnicas de segmentación
```

## Técnicas Implementadas

### 1. Transformaciones Geométricas ✅

#### Rotación
- Rotación de 45° alrededor del centro
- Usa `cv2.getRotationMatrix2D()` y `cv2.warpAffine()`

#### Escalado
- Escalado 1.5x con interpolación lineal
- Recorte al tamaño original para comparación

#### Traslación
- Desplazamiento de 100px derecha, 50px abajo
- Matriz de transformación afín

#### Transformación Afín
- Mapeo de 3 puntos a nuevas posiciones
- Permite rotación, escalado y traslación combinados

#### Reflexión
- Espejo horizontal usando `cv2.flip()`

### 2. Ajuste de Contraste y Brillo ✅

#### Ajuste Lineal
- **Brillo**: `new_pixel = pixel + beta` (beta=50)
- **Contraste**: `new_pixel = alpha * pixel` (alpha=1.5)
- Función: `cv2.convertScaleAbs()`

#### Ecualización de Histograma
- Redistribuye intensidades para mejorar contraste
- `cv2.equalizeHist()` en escala de grises

#### CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Ecualización adaptativa por regiones
- Evita sobre-amplificación de ruido
- Parámetros: `clipLimit=2.0`, `tileGridSize=(8,8)`

#### Corrección Gamma
- **Gamma < 1**: Aumenta brillo (gamma=0.5)
- **Gamma > 1**: Reduce brillo (gamma=2.0)
- Transformación no lineal: `output = (input/255)^(1/gamma) * 255`

### 3. Filtros Espaciales ✅

#### Filtro Gaussiano
- Suavizado mediante convolución con kernel gaussiano
- Reduce ruido gaussiano
- Kernel: 15x15

#### Filtro de Mediana
- Elimina ruido sal y pimienta
- Reemplaza pixel por mediana de vecindario
- Kernel: 9x9

#### Filtro Bilateral
- Suaviza preservando bordes
- Considera distancia espacial Y similitud de intensidad
- Parámetros: `d=9`, `sigmaColor=75`, `sigmaSpace=75`

#### Filtro de Nitidez (Sharpening)
- Realza bordes y detalles
- Kernel: `[[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]`

#### Filtro Laplaciano
- Detección de bordes basada en segunda derivada
- Resalta cambios rápidos de intensidad

#### Filtro de Promedio
- Suavizado simple por promedio de vecindario
- Kernel: 15x15

#### Filtro Sobel
- Detección de gradientes (bordes direccionales)
- Calcula derivadas en X e Y
- Magnitud: `sqrt(Gx² + Gy²)`

#### Unsharp Masking
- Realce de detalles finos
- Fórmula: `sharp = original + (original - blur) * amount`
- Parámetros: `amount=1.5`

### 4. Segmentación de Imágenes ✅

#### Umbralización Simple
- Binarización con umbral fijo (127)
- `cv2.threshold(THRESH_BINARY)`

#### Umbralización Otsu
- Calcula umbral óptimo automáticamente
- Maximiza varianza entre clases
- `cv2.threshold(THRESH_OTSU)`

#### Umbralización Adaptativa
- Umbral diferente por región
- Método gaussiano con ventana 11x11
- Útil para iluminación no uniforme

#### Watershed (Segmentación basada en regiones)
- Trata imagen como topografía
- Identifica regiones por "inundación"
- Pasos:
  1. Transformada de distancia
  2. Marcadores de primer plano/fondo
  3. Algoritmo watershed

#### K-means (Agrupamiento)
- Agrupa píxeles por similitud de color
- **K=3**: Segmentación simple (fondo, mano, sombra)
- **K=5**: Segmentación más detallada
- Criterio: 100 iteraciones o convergencia

#### GrabCut (Segmentación interactiva)
- Segmentación foreground/background
- Usa modelo de mezcla de gaussianas (GMM)
- Rectángulo inicial: 10% margen desde bordes
- 5 iteraciones de refinamiento

## Uso

### Opción 1: Ejecutar todo desde notebook

```python
# Abrir Preprocesamiento_Completo.ipynb
%run tecnicas_procesamiento.py
```

### Opción 2: Ejecutar técnicas individuales

```python
from tecnicas_procesamiento import (
    transformaciones_geometricas,
    ajuste_contraste_brillo,
    filtros_espaciales,
    segmentacion_imagenes
)

# Procesar una imagen específica
imagen = 'data/inputs/mano-abierta.jpg'
transformaciones_geometricas(imagen)
ajuste_contraste_brillo(imagen)
filtros_espaciales(imagen)
segmentacion_imagenes(imagen)
```

### Opción 3: Ejecutar desde terminal

```bash
cd /home/yep/Documentos/visionArtificial/Proyecto-Gestos-de-la-mano-Visi-n-Artificial
python tecnicas_procesamiento.py
```

## Resultados

Cada técnica genera:
- **Imágenes procesadas** guardadas en subcarpetas organizadas
- **Visualización comparativa** con matplotlib (3x3 grid)
- **Imagen de comparación** guardada como `comparacion.jpg`

### Estructura de salida

```
data/procesadas/
├── transformaciones/
│   ├── mano-abierta/
│   │   ├── rotacion_45.jpg
│   │   ├── escalado_1.5x.jpg
│   │   ├── traslacion.jpg
│   │   ├── afin.jpg
│   │   ├── reflexion_horizontal.jpg
│   │   └── comparacion.jpg
│   ├── mano-cerrada/
│   └── mano-pulgar/
├── contraste_brillo/
│   ├── mano-abierta/
│   │   ├── brillo_aumentado.jpg
│   │   ├── contraste_aumentado.jpg
│   │   ├── ecualizacion_histograma.jpg
│   │   ├── clahe.jpg
│   │   ├── gamma_0.5.jpg
│   │   ├── gamma_2.0.jpg
│   │   └── comparacion.jpg
│   └── ...
├── filtros_espaciales/
│   ├── mano-abierta/
│   │   ├── gaussiano.jpg
│   │   ├── mediana.jpg
│   │   ├── bilateral.jpg
│   │   ├── nitidez.jpg
│   │   ├── laplaciano.jpg
│   │   ├── promedio.jpg
│   │   ├── sobel.jpg
│   │   ├── unsharp_mask.jpg
│   │   └── comparacion.jpg
│   └── ...
└── segmentacion/
    ├── mano-abierta/
    │   ├── umbral_simple.jpg
    │   ├── umbral_otsu.jpg
    │   ├── umbral_adaptativo.jpg
    │   ├── watershed.jpg
    │   ├── kmeans_k3.jpg
    │   ├── kmeans_k5.jpg
    │   ├── grabcut.jpg
    │   └── comparacion.jpg
    └── ...
```

## Dependencias

```python
import cv2          # OpenCV para procesamiento
import numpy as np  # Operaciones numéricas
import matplotlib.pyplot as plt  # Visualización
import os           # Manejo de archivos
```

## Parámetros Configurables

En `tecnicas_procesamiento.py`:

```python
TARGET_SIZE = (800, 800)  # Tamaño uniforme de imágenes
IMAGENES = [              # Lista de imágenes a procesar
    'data/inputs/mano-abierta.jpg',
    'data/inputs/mano-cerrada.jpg',
    'data/inputs/mano-pulgar.png'
]
```

## Notas Técnicas

### Mejores prácticas implementadas:
- ✅ Todas las imágenes se redimensionan a 800x800 para consistencia
- ✅ Resultados organizados por técnica y por imagen
- ✅ Visualizaciones comparativas automáticas
- ✅ Código modular y reutilizable
- ✅ Manejo de errores para imágenes faltantes

### Consideraciones:
- **Watershed** puede ser sensible a ruido → se aplica filtro mediana primero
- **K-means** es computacionalmente intensivo → puede tomar tiempo
- **GrabCut** requiere rectángulo inicial → se usa 80% del área central
- **CLAHE** es mejor que ecualización global para iluminación no uniforme

## Aplicación al Proyecto de Gestos

Estas técnicas son útiles para:

1. **Data augmentation** (transformaciones geométricas)
2. **Normalización** (ajuste de contraste/brillo)
3. **Reducción de ruido** (filtros espaciales)
4. **Extracción de mano** (segmentación)
5. **Detección de contornos** (Canny, Sobel, Laplaciano)

## Referencias

- OpenCV Documentation: https://docs.opencv.org/
- Digital Image Processing (Gonzalez & Woods)
- Watershed Algorithm: Vincent & Soille (1991)
- GrabCut: Rother et al. (2004)
- CLAHE: Zuiderveld (1994)
