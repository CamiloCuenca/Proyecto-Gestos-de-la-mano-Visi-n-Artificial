import pandas as pd

# Ruta del archivo original
ruta_csv = "data/caracteristicas/gestos_dataset.csv"

# Cargar el dataset (saltando líneas malas)
df = pd.read_csv(ruta_csv, on_bad_lines='skip', encoding='utf-8')

# Mantener solo las columnas relevantes
columnas_validas = [
    'ORB_mean', 'ORB_std', 'ORB_entropy',
    'SIFT_mean', 'SIFT_std', 'SIFT_entropy',
    'AKAZE_mean', 'AKAZE_std', 'AKAZE_entropy',
    'label'
]

# Filtrar columnas válidas y eliminar filas con datos faltantes
df_clean = df[columnas_validas].dropna().reset_index(drop=True)

# Sobrescribir el archivo original
df_clean.to_csv(ruta_csv, index=False, encoding='utf-8')

# Confirmación visual
print("✅ Dataset limpiado y sobrescrito correctamente.")
print(f"🔢 Filas: {df_clean.shape[0]}, Columnas: {df_clean.shape[1]}")
print("\nDistribución de etiquetas:")
print(df_clean['label'].value_counts())

