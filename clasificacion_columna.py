# Importar las bibliotecas necesarias
from transformers import pipeline
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Cargar los datos
df = pd.read_csv('Reviews.csv')

# Asegurar que no hay valores nulos en las columnas importantes
df = df.dropna(subset=['Summary', 'Text', 'Score', 'UserId'])

# 2. Definir las etiquetas comunes (ejemplo)
common_labels = [
    'electronics', 'clothing', 'books', 'home', 'toys', 'kitchen', 'sports', 'health', 'automotive',
    'beauty', 'office', 'music', 'grocery', 'computers', 'furniture', 'appliances', 'video games',
    'baby', 'jewelry', 'shoes', 'tools', 'outdoors', 'software', 'industrial', 'pet supplies',
    'hardware', 'gift cards', 'luggage', 'movies', 'digital music', 'personal care', 'watches',
    'blended', 'sunglasses', 'mattresses', 'backpacks', 'yoga', 'accessories', 'cds', 'vhs', 'puzzles',
    'fishing', 'cameras', 'car electronics', 'bike', 'drones', 'air conditioners', 'heaters', 'lawn',
    'safety', 'decor', 'kitchen tools', 'travel'
]

# 3. Codificar las etiquetas
label_encoder = LabelEncoder()
label_encoder.fit(common_labels)

# 4. Configurar el pipeline de clasificación zero-shot
zero_shot_classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

# 5. Parámetros de procesamiento por lotes
batch_size = 1  # Tamaño del lote (puedes ajustarlo según la memoria disponible)
output_file = 'Reviews_etiquetado.csv'

# Si el archivo de salida no existe, créalo e incluye el encabezado
try:
    open(output_file, 'x').close()  # Crea el archivo si no existe
    df[:0].to_csv(output_file, index=False)  # Guarda solo el encabezado
except FileExistsError:
    pass

# 6. Función para aplicar clasificación zero-shot en lotes
def process_batch(batch_df):
    # Realiza la clasificación zero-shot para cada fila en el lote
    results = zero_shot_classifier(batch_df['Summary'].tolist(), candidate_labels=common_labels)
    # Extrae la etiqueta con la mayor puntuación para cada fila
    labels = [result['labels'][0] for result in results]
    return labels

# 7. Procesamiento por lotes
for start in range(0, len(df), batch_size):
    end = min(start + batch_size, len(df))
    batch_df = df.iloc[start:end].copy()  # Extraer un lote de filas

    # Aplicar la clasificación zero-shot al lote
    print(f"Procesando filas {start} a {end}")
    batch_df['zero_shot_label'] = process_batch(batch_df)

    # Guardar los resultados en el archivo CSV de salida
    batch_df.to_csv(output_file, mode='a', index=False, header=False)  # Agregar sin sobrescribir ni duplicar el encabezado

print("Procesamiento completado.")
