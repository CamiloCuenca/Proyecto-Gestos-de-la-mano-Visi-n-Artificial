"""inspect_clean.py
Script auxiliar para cargar `data/caracteristicas/gestos_dataset.csv`, limpiar
con las columnas numéricas de descriptores y mostrar un resumen en consola.

Uso: python inspect_clean.py
"""
import pandas as pd
import os

CSV_PATH = os.path.join('data', 'caracteristicas', 'gestos_dataset.csv')

desired_cols = ['ORB_mean', 'ORB_std', 'ORB_entropy',
                'SIFT_mean', 'SIFT_std', 'SIFT_entropy',
                'AKAZE_mean', 'AKAZE_std', 'AKAZE_entropy',
                'label']

def clean_colname(name):
    import re
    name = re.sub(r"\s+", "_", str(name))
    name = re.sub(r"[^0-9A-Za-z_]+", "", name)
    return name

def main():
    if not os.path.exists(CSV_PATH):
        print('No se encontró', CSV_PATH)
        return

    print('Leyendo:', CSV_PATH)
    df = pd.read_csv(CSV_PATH, on_bad_lines='skip', encoding='utf-8')
    # normalizar nombres
    df.columns = [clean_colname(c) for c in df.columns]
    available = [c for c in desired_cols if c in df.columns]
    missing = [c for c in desired_cols if c not in df.columns]
    print('Columnas disponibles para limpieza:', available)
    if missing:
        print('Columnas esperadas faltantes:', missing)

    if not available:
        print('No hay columnas disponibles entre las deseadas. Abortando.')
        return

    df_clean = df[available].copy()
    for c in available:
        if c != 'label':
            df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')

    before = df_clean.shape[0]
    df_clean = df_clean.dropna()
    after = df_clean.shape[0]
    print(f'Filas antes de dropna: {before}  después: {after}')
    print('Dimensiones df_clean:', df_clean.shape)
    print('\nConteo de clases:')
    print(df_clean['label'].value_counts(dropna=False))
    print('\nPrimeras filas de df_clean:')
    print(df_clean.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
