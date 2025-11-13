"""generate_gestos_dataset.py
Genera `data/caracteristicas/gestos_dataset.csv` combinando `metrics.csv`
con resúmenes estadísticos de descriptores (.npy) en
`data/caracteristicas/descriptores/<label>/`.

Uso: python generate_gestos_dataset.py

Salida: data/caracteristicas/gestos_dataset.csv
"""
import os
import glob
import re
import numpy as np
import pandas as pd
from collections import defaultdict


def clean_colname(name):
    # Mantener solo letras, números y guiones bajos
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9A-Za-z_]+", "", name)
    return name


def compute_entropy(arr, bins=256):
    # arr: 1D array
    if arr.size == 0:
        return np.nan
    try:
        hist, _ = np.histogram(arr, bins=bins)
        probs = hist.astype(float) / hist.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy)
    except Exception:
        return np.nan


def summarize_descriptor(path):
    """Carga un .npy y devuelve (mean, std, entropy) sobre todos los valores."""
    try:
        arr = np.load(path, allow_pickle=True)
        # Acomodar distintos formatos: lista de arrays, 2D array, etc.
        if isinstance(arr, (list, tuple, np.ndarray)) and len(arr) == 0:
            flat = np.array([])
        else:
            try:
                flat = np.asarray(arr).astype(float).ravel()
            except Exception:
                # fallback: try to flatten pickled objects
                flat = np.hstack([np.ravel(x).astype(float) for x in arr if x is not None])
        if flat.size == 0:
            return {"mean": np.nan, "std": np.nan, "entropy": np.nan}
        return {"mean": float(np.mean(flat)), "std": float(np.std(flat)), "entropy": compute_entropy(flat)}
    except Exception:
        return {"mean": np.nan, "std": np.nan, "entropy": np.nan}


def load_descriptor_summaries(descriptors_root):
    """Recorre subcarpetas y resume akaze/orb/sift por carpeta (label)."""
    summaries = {}
    if not os.path.isdir(descriptors_root):
        return summaries
    for folder in sorted(os.listdir(descriptors_root)):
        fpath = os.path.join(descriptors_root, folder)
        if not os.path.isdir(fpath):
            continue
        label = folder
        stats = {}
        # archivos esperados
        mapping = {"AKAZE": "akaze_descriptors.npy", "ORB": "orb_descriptors.npy", "SIFT": "sift_descriptors.npy"}
        for key, fname in mapping.items():
            p = os.path.join(fpath, fname)
            if os.path.exists(p):
                s = summarize_descriptor(p)
                stats[f"{key}_mean"] = s.get("mean")
                stats[f"{key}_std"] = s.get("std")
                stats[f"{key}_entropy"] = s.get("entropy")
            else:
                stats[f"{key}_mean"] = np.nan
                stats[f"{key}_std"] = np.nan
                stats[f"{key}_entropy"] = np.nan
        summaries[label] = stats
    return summaries


def assign_label_from_image(img_name, descriptor_labels):
    """Intenta inferir label a partir del nombre de la imagen.
    descriptor_labels: lista de folder names
    """
    if not isinstance(img_name, str):
        return "unknown"
    s = img_name.lower()
    # buscar coincidencia por contención
    for lab in descriptor_labels:
        if lab.lower() in s:
            return lab
    # fallback: tomar parte antes del primer '_' si existe
    base = os.path.splitext(os.path.basename(img_name))[0]
    if "_" in base:
        candidate = base.split("_")[0]
        for lab in descriptor_labels:
            if candidate.lower() in lab.lower() or lab.lower() in candidate.lower():
                return lab
        return candidate
    return "unknown"


def main():
    root = os.path.join("data", "caracteristicas")
    metrics_csv = os.path.join(root, "metrics.csv")
    out_csv = os.path.join(root, "gestos_dataset.csv")
    descriptors_root = os.path.join(root, "descriptores")

    # Cargar métricas: preferir metrics.csv si existe, si no buscar metricas.txt en subcarpetas
    if os.path.exists(metrics_csv):
        print(f"Usando archivo de métricas: {metrics_csv}")
        df = pd.read_csv(metrics_csv)
    else:
        # Buscar todos los metricas.txt en subcarpetas y combinarlos
        txt_paths = glob.glob(os.path.join(root, '**', 'metricas.txt'), recursive=True)
        print("No se encontró 'metrics.csv'. Buscando 'metricas.txt' en subcarpetas...")
        if txt_paths:
            print("Se encontraron los siguientes archivos 'metricas.txt':")
            for p in txt_paths:
                print(" -", p)
        else:
            print("No se encontraron archivos 'metricas.txt' en las subcarpetas.")
        rows = []
        for p in txt_paths:
            try:
                # intentar leer como CSV simple
                try:
                    tmp = pd.read_csv(p, sep=None, engine='python')
                    # si tmp tiene más de 1 column y la primera fila parece ser header, convertir a dict
                    if tmp.shape[1] == 1:
                        # formateado como key:value en una columna
                        ser = tmp[tmp.columns[0]].astype(str).tolist()
                        d = {}
                        for line in ser:
                            if ':' in line:
                                k, v = line.split(':', 1)
                                d[k.strip()] = v.strip()
                            elif '=' in line:
                                k, v = line.split('=', 1)
                                d[k.strip()] = v.strip()
                        row = d
                    else:
                        # convertir fila 0 a dict si es un archivo con una fila de valores
                        if tmp.shape[0] == 1:
                            row = dict(zip(tmp.columns.astype(str), tmp.iloc[0].values.astype(str)))
                        else:
                            # tomar la primera fila como ejemplo
                            row = dict(zip(tmp.columns.astype(str), tmp.iloc[0].values.astype(str)))
                except Exception:
                    # fallback: parsear manualmente con soporte para secciones
                    row = {}
                    with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
                        current_section = None
                        for raw in fh:
                            line = raw.rstrip('\n')
                            s = line.strip()
                            if not s:
                                continue
                            # detectar encabezado de sección (p.ej. 'SIFT:')
                            m = re.match(r'^([A-Za-z0-9_\- ]+):\s*$', s)
                            if m and (s.endswith(':') and not ':' in s[:-1]):
                                current_section = m.group(1).strip()
                                continue
                            # líneas con guion '-' (ítems) o con formato clave: valor
                            # quitar guion inicial si existe
                            if s.startswith('-'):
                                s = s.lstrip('-').strip()
                            # ahora intentar split clave:valor
                            if ':' in s:
                                k, v = s.split(':', 1)
                                key = k.strip()
                                val = v.strip()
                            elif '=' in s:
                                k, v = s.split('=', 1)
                                key = k.strip()
                                val = v.strip()
                            else:
                                # línea suelta -> usar como value genérico
                                key = 'value'
                                val = s

                            # si estamos dentro de una sección, prefijar la clave
                            if current_section:
                                pref = f"{current_section}_{key}"
                            else:
                                pref = key

                            # Normalizar y convertir valores numéricos o tuplas
                            # detectar tuplas como '(a, b)'
                            tup_match = re.match(r"^\((\s*\d+)\s*,\s*(\d+)\s*\)$", val)
                            if tup_match:
                                r_val = int(tup_match.group(1))
                                c_val = int(tup_match.group(2))
                                row[f"{pref}_rows"] = r_val
                                row[f"{pref}_cols"] = c_val
                            else:
                                # intentar convertir a int o float
                                val_clean = val.replace(',', '')
                                try:
                                    if '.' in val_clean:
                                        num = float(val_clean)
                                    else:
                                        num = int(val_clean)
                                    row[pref] = num
                                except Exception:
                                    row[pref] = val

                # metadata: label y grupo (padre directo bajo data/caracteristicas)
                rel = os.path.relpath(p, root)
                parts = rel.split(os.sep)
                group = parts[0] if len(parts) > 1 else ''
                label = parts[1] if len(parts) > 1 else os.path.splitext(parts[-1])[0]
                row['__source_path'] = p
                row['__group'] = group
                row['Imagen'] = label
                rows.append(row)
            except Exception:
                continue
        if not rows:
            print(f"ERROR: no se encontraron archivos 'metricas.txt' bajo {root} y tampoco existe {metrics_csv}")
            return
        df = pd.DataFrame(rows)
    # Normalizar nombres de columnas (excepto metadata internamente añadida)
    orig_cols = list(df.columns)
    rename_map = {c: clean_colname(c) for c in orig_cols}
    df = df.rename(columns=rename_map)

    # Resumir descriptores por carpeta
    summaries = load_descriptor_summaries(descriptors_root)
    labels = list(summaries.keys())
    if summaries:
        print("Carpetas de descriptores detectadas:")
        for lab in labels:
            print(f" - {lab}: {os.path.join(descriptors_root, lab)}")
    else:
        print(f"No se encontraron carpetas de descriptores en {descriptors_root}")

    # Columnas descriptor a agregar (en orden): ORB_mean, ORB_std, SIFT_mean, SIFT_std, AKAZE_mean, AKAZE_std
    # También incluimos entropies como columnas adicionales
    descriptor_cols = []
    for key in ["ORB", "SIFT", "AKAZE"]:
        descriptor_cols.append(f"{key}_mean")
        descriptor_cols.append(f"{key}_std")
        descriptor_cols.append(f"{key}_entropy")

    # Crear columnas vacías
    for col in descriptor_cols:
        df[col] = np.nan

    # Asignar label y rellenar columnas
    df['label'] = 'unknown'
    possible_cols = [c for c in df.columns if 'imagen' in c.lower() or 'image' in c.lower() or 'file' in c.lower()]
    for idx, row in df.iterrows():
        imagen = None
        if possible_cols:
            imagen = row[possible_cols[0]]
        else:
            # si se creó desde metricas.txt usamos la columna Imagen si existe
            if 'Imagen' in df.columns:
                imagen = row['Imagen']
            else:
                imagen = None

        label = assign_label_from_image(str(imagen) if imagen is not None else "", labels)
        # si no encontramos label, intentar con columna __group (p.ej. 'bordes', 'descriptores', 'formas') y nombre Imagen
        if label == 'unknown' and '__group' in df.columns:
            grp = row.get('__group', '')
            # si group está en summaries keys, usarlo
            if grp in summaries:
                label = grp
        df.at[idx, 'label'] = label
        # rellenar descriptor stats
        if label in summaries:
            stats = summaries[label]
            for k, v in stats.items():
                df.at[idx, k] = v

    # Reordenar columnas para ser amigables con MATLAB
    final_cols = list(df.columns)
    # asegurar que descriptor columns estén presentes
    for col in descriptor_cols:
        if col not in final_cols:
            final_cols.append(col)
    # mover label al final
    if 'label' in final_cols:
        final_cols = [c for c in final_cols if c != 'label'] + ['label']

    df_out = df[final_cols]
    # Guardar
    os.makedirs(root, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    # Resumen
    print("Dataset generado:", out_csv)
    print("Filas:", df_out.shape[0], "Columnas:", df_out.shape[1])
    print("Clases encontradas:")
    print(df_out['label'].value_counts(dropna=False))

    # Opcional: gráfico de distribución de clases si matplotlib está disponible
    try:
        import matplotlib.pyplot as plt
        vc = df_out['label'].value_counts()
        plt.figure(figsize=(6,4))
        vc.plot(kind='bar')
        plt.title('Distribución de clases')
        plt.xlabel('label')
        plt.ylabel('count')
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

    # -----------------------------
    # Limpieza final en memoria (df_clean)
    # Cargar el CSV generado usando on_bad_lines='skip' y encoding utf-8
    # Conservar solo columnas numéricas de descriptores + label
    # -----------------------------
    desired_cols = ['ORB_mean', 'ORB_std', 'ORB_entropy',
                    'SIFT_mean', 'SIFT_std', 'SIFT_entropy',
                    'AKAZE_mean', 'AKAZE_std', 'AKAZE_entropy',
                    'label']

    try:
        df_loaded = pd.read_csv(out_csv, on_bad_lines='skip', encoding='utf-8')
        print(f"Loaded CSV for cleaning: {out_csv}")
    except Exception as e:
        print(f"Advertencia: no se pudo leer {out_csv} con pandas.read_csv ({e}), usando DataFrame en memoria.")
        df_loaded = df_out.copy()

    # Normalizar nombres de columnas en df_loaded (por si hay espacios u otros caracteres)
    df_loaded.columns = [clean_colname(c) for c in df_loaded.columns]

    # Seleccionar solo las columnas disponibles entre las deseadas
    available = [c for c in desired_cols if c in df_loaded.columns]
    missing = [c for c in desired_cols if c not in df_loaded.columns]
    if missing:
        print("Columnas esperadas faltantes en el CSV:", missing)
    if not available:
        print("ERROR: Ninguna de las columnas deseadas está disponible. df_clean no se creó.")
        df_clean = pd.DataFrame()
    else:
        df_clean = df_loaded[available].copy()
        # Forzar conversión a numéricas para las columnas de descriptors
        for c in available:
            if c != 'label':
                df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')

        # Eliminar filas con NaN (valores vacíos o corruptos)
        before = df_clean.shape[0]
        df_clean = df_clean.dropna()
        after = df_clean.shape[0]
        print(f"Filas antes de dropna: {before}, después de dropna: {after}")

        # Mostrar resumen final
        print("DataFrame limpio en memoria: df_clean")
        print("Filas:", df_clean.shape[0], "Columnas:", df_clean.shape[1])
        print("Conteo de clases (label):")
        print(df_clean['label'].value_counts(dropna=False))


if __name__ == '__main__':
    main()
