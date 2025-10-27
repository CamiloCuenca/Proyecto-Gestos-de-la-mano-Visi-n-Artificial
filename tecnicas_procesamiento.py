"""
Técnicas Avanzadas de Procesamiento de Imágenes
Complementa el preprocesamiento básico con:
- Transformaciones geométricas
- Ajuste de contraste y brillo  
- Filtros espaciales
- Segmentación avanzada
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuración
IMAGENES = ['data/inputs/mano-abierta.jpg', 'data/inputs/mano-cerrada.jpg', 'data/inputs/mano-pulgar.png']
TARGET_SIZE = (800, 800)

def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    """alpha: contraste (1.0-3.0), beta: brillo (0-100)"""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def adjust_gamma(img, gamma=1.0):
    """Corrección gamma"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def transformaciones_geometricas(imagen_path):
    """Aplica transformaciones geométricas"""
    img_name = os.path.splitext(os.path.basename(imagen_path))[0]
    print(f"\n{'='*50}\nTransformaciones geométricas: {img_name}\n{'='*50}")
    
    img = cv2.imread(imagen_path)
    if img is None:
        print(f"Error: No se pudo leer {imagen_path}")
        return
    
    img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    height, width = img_resized.shape[:2]
    
    output_dir = f'data/procesadas/transformaciones/{img_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Rotación 45°
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    img_rotated = cv2.warpAffine(img_resized, rotation_matrix, (width, height))
    cv2.imwrite(f"{output_dir}/rotacion_45.jpg", img_rotated)
    
    # 2. Escalado 1.5x
    img_scaled = cv2.resize(img_resized, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    h, w = img_scaled.shape[:2]
    start_h, start_w = (h - height) // 2, (w - width) // 2
    img_scaled_crop = img_scaled[start_h:start_h+height, start_w:start_w+width]
    cv2.imwrite(f"{output_dir}/escalado_1.5x.jpg", img_scaled_crop)
    
    # 3. Traslación
    translation_matrix = np.float32([[1, 0, 100], [0, 1, 50]])
    img_translated = cv2.warpAffine(img_resized, translation_matrix, (width, height))
    cv2.imwrite(f"{output_dir}/traslacion.jpg", img_translated)
    
    # 4. Transformación afín
    pts1 = np.float32([[50, 50], [width-50, 50], [50, height-50]])
    pts2 = np.float32([[100, 100], [width-100, 80], [80, height-100]])
    affine_matrix = cv2.getAffineTransform(pts1, pts2)
    img_affine = cv2.warpAffine(img_resized, affine_matrix, (width, height))
    cv2.imwrite(f"{output_dir}/afin.jpg", img_affine)
    
    # 5. Reflexión horizontal
    img_flip = cv2.flip(img_resized, 1)
    cv2.imwrite(f"{output_dir}/reflexion_horizontal.jpg", img_flip)
    
    # Visualización
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Rotación 45°')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(img_scaled_crop, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Escalado 1.5x')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(img_translated, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Traslación')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(img_affine, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Transformación Afín')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cv2.cvtColor(img_flip, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Reflexión Horizontal')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparacion.jpg", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Guardado en: {output_dir}/")

def ajuste_contraste_brillo(imagen_path):
    """Ajusta contraste y brillo"""
    img_name = os.path.splitext(os.path.basename(imagen_path))[0]
    print(f"\n{'='*50}\nAjuste contraste/brillo: {img_name}\n{'='*50}")
    
    img = cv2.imread(imagen_path)
    if img is None:
        print(f"Error: No se pudo leer {imagen_path}")
        return
    
    img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    output_dir = f'data/procesadas/contraste_brillo/{img_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Ajuste lineal - brillo
    img_bright = adjust_brightness_contrast(img_resized, alpha=1.0, beta=50)
    cv2.imwrite(f"{output_dir}/brillo_aumentado.jpg", img_bright)
    
    # 2. Ajuste lineal - contraste
    img_contrast = adjust_brightness_contrast(img_resized, alpha=1.5, beta=0)
    cv2.imwrite(f"{output_dir}/contraste_aumentado.jpg", img_contrast)
    
    # 3. Ecualización de histograma
    img_eq = cv2.equalizeHist(gray)
    cv2.imwrite(f"{output_dir}/ecualizacion_histograma.jpg", img_eq)
    
    # 4. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(gray)
    cv2.imwrite(f"{output_dir}/clahe.jpg", img_clahe)
    
    # 5. Corrección gamma
    img_gamma_bright = adjust_gamma(gray, gamma=0.5)
    img_gamma_dark = adjust_gamma(gray, gamma=2.0)
    cv2.imwrite(f"{output_dir}/gamma_0.5.jpg", img_gamma_bright)
    cv2.imwrite(f"{output_dir}/gamma_2.0.jpg", img_gamma_dark)
    
    # Visualización
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    axes[0, 0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(img_bright, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Brillo +50')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(img_contrast, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Contraste x1.5')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(gray, cmap='gray')
    axes[1, 0].set_title('Escala de grises')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_eq, cmap='gray')
    axes[1, 1].set_title('Ecualización Histograma')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(img_clahe, cmap='gray')
    axes[1, 2].set_title('CLAHE')
    axes[1, 2].axis('off')
    
    axes[2, 0].imshow(img_gamma_bright, cmap='gray')
    axes[2, 0].set_title('Gamma 0.5')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(img_gamma_dark, cmap='gray')
    axes[2, 1].set_title('Gamma 2.0')
    axes[2, 1].axis('off')
    
    axes[2, 2].hist(gray.ravel(), 256, [0, 256], color='blue', alpha=0.5, label='Original')
    axes[2, 2].hist(img_eq.ravel(), 256, [0, 256], color='red', alpha=0.5, label='Ecualizado')
    axes[2, 2].set_title('Histogramas')
    axes[2, 2].legend()
    axes[2, 2].set_xlim([0, 256])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparacion.jpg", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Guardado en: {output_dir}/")

def filtros_espaciales(imagen_path):
    """Aplica filtros espaciales"""
    img_name = os.path.splitext(os.path.basename(imagen_path))[0]
    print(f"\n{'='*50}\nFiltros espaciales: {img_name}\n{'='*50}")
    
    img = cv2.imread(imagen_path)
    if img is None:
        print(f"Error: No se pudo leer {imagen_path}")
        return
    
    img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    output_dir = f'data/procesadas/filtros_espaciales/{img_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Filtro Gaussiano
    img_gaussian = cv2.GaussianBlur(img_resized, (15, 15), 0)
    cv2.imwrite(f"{output_dir}/gaussiano.jpg", img_gaussian)
    
    # 2. Filtro de Mediana
    img_median = cv2.medianBlur(img_resized, 9)
    cv2.imwrite(f"{output_dir}/mediana.jpg", img_median)
    
    # 3. Filtro Bilateral
    img_bilateral = cv2.bilateralFilter(img_resized, 9, 75, 75)
    cv2.imwrite(f"{output_dir}/bilateral.jpg", img_bilateral)
    
    # 4. Filtro de Nitidez
    kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_sharpen = cv2.filter2D(img_resized, -1, kernel_sharpen)
    cv2.imwrite(f"{output_dir}/nitidez.jpg", img_sharpen)
    
    # 5. Filtro Laplaciano
    img_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    img_laplacian = cv2.convertScaleAbs(img_laplacian)
    cv2.imwrite(f"{output_dir}/laplaciano.jpg", img_laplacian)
    
    # 6. Filtro de Promedio
    img_average = cv2.blur(img_resized, (15, 15))
    cv2.imwrite(f"{output_dir}/promedio.jpg", img_average)
    
    # 7. Filtro Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    img_sobel = cv2.magnitude(sobelx, sobely)
    img_sobel = cv2.convertScaleAbs(img_sobel)
    cv2.imwrite(f"{output_dir}/sobel.jpg", img_sobel)
    
    # 8. Unsharp Masking
    gaussian = cv2.GaussianBlur(img_resized, (9, 9), 10.0)
    img_unsharp = cv2.addWeighted(img_resized, 1.5, gaussian, -0.5, 0)
    cv2.imwrite(f"{output_dir}/unsharp_mask.jpg", img_unsharp)
    
    # Visualización
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    axes[0, 0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(img_gaussian, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Gaussiano')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(img_median, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Mediana')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(img_bilateral, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Bilateral')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(img_sharpen, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Nitidez')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(img_laplacian, cmap='gray')
    axes[1, 2].set_title('Laplaciano')
    axes[1, 2].axis('off')
    
    axes[2, 0].imshow(cv2.cvtColor(img_average, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title('Promedio')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(img_sobel, cmap='gray')
    axes[2, 1].set_title('Sobel')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(cv2.cvtColor(img_unsharp, cv2.COLOR_BGR2RGB))
    axes[2, 2].set_title('Unsharp Masking')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparacion.jpg", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Guardado en: {output_dir}/")

def segmentacion_imagenes(imagen_path):
    """Segmentación de imágenes"""
    img_name = os.path.splitext(os.path.basename(imagen_path))[0]
    print(f"\n{'='*50}\nSegmentación: {img_name}\n{'='*50}")
    
    img = cv2.imread(imagen_path)
    if img is None:
        print(f"Error: No se pudo leer {imagen_path}")
        return
    
    img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    output_dir = f'data/procesadas/segmentacion/{img_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Umbralización simple
    _, thresh_simple = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"{output_dir}/umbral_simple.jpg", thresh_simple)
    
    # 2. Umbralización Otsu
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f"{output_dir}/umbral_otsu.jpg", thresh_otsu)
    
    # 3. Umbralización adaptativa
    thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(f"{output_dir}/umbral_adaptativo.jpg", thresh_adaptive)
    
    # 4. Watershed
    blur = cv2.medianBlur(gray, 5)
    _, thresh_ws = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh_ws, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    img_watershed = img_resized.copy()
    markers = cv2.watershed(img_watershed, markers)
    img_watershed[markers == -1] = [0, 0, 255]
    cv2.imwrite(f"{output_dir}/watershed.jpg", img_watershed)
    
    # 5. K-means (K=3)
    pixel_values = img_resized.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_img = centers[labels.flatten()].reshape(img_resized.shape)
    cv2.imwrite(f"{output_dir}/kmeans_k3.jpg", segmented_img)
    
    # 6. K-means (K=5)
    _, labels5, centers5 = cv2.kmeans(pixel_values, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers5 = np.uint8(centers5)
    segmented_img5 = centers5[labels5.flatten()].reshape(img_resized.shape)
    cv2.imwrite(f"{output_dir}/kmeans_k5.jpg", segmented_img5)
    
    # 7. GrabCut
    mask_grabcut = np.zeros(img_resized.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    height, width = img_resized.shape[:2]
    rect = (int(width*0.1), int(height*0.1), int(width*0.8), int(height*0.8))
    cv2.grabCut(img_resized, mask_grabcut, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask_grabcut == 2) | (mask_grabcut == 0), 0, 1).astype('uint8')
    img_grabcut = img_resized * mask2[:, :, np.newaxis]
    cv2.imwrite(f"{output_dir}/grabcut.jpg", img_grabcut)
    
    # Visualización
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    axes[0, 0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(thresh_simple, cmap='gray')
    axes[0, 1].set_title('Umbral Simple')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(thresh_otsu, cmap='gray')
    axes[0, 2].set_title('Umbral Otsu')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(thresh_adaptive, cmap='gray')
    axes[1, 0].set_title('Umbral Adaptativo')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(img_watershed, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Watershed')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('K-means (K=3)')
    axes[1, 2].axis('off')
    
    axes[2, 0].imshow(cv2.cvtColor(segmented_img5, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title('K-means (K=5)')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(cv2.cvtColor(img_grabcut, cv2.COLOR_BGR2RGB))
    axes[2, 1].set_title('GrabCut')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(dist_transform, cmap='hot')
    axes[2, 2].set_title('Transformada Distancia')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparacion.jpg", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Guardado en: {output_dir}/")

if __name__ == "__main__":
    print("="*60)
    print("TÉCNICAS AVANZADAS DE PROCESAMIENTO DE IMÁGENES")
    print("="*60)
    
    for imagen in IMAGENES:
        transformaciones_geometricas(imagen)
        ajuste_contraste_brillo(imagen)
        filtros_espaciales(imagen)
        segmentacion_imagenes(imagen)
    
    print("\n" + "="*60)
    print("¡PROCESAMIENTO COMPLETADO!")
    print("="*60)
