from transformers import pipeline
import pandas as pd

# Cargar datos
df = pd.read_csv('Reviews.csv')
df = df.dropna(subset=['Summary', 'Text'])

# Inicializar contador global
contador = 0

# Cargar el modelo de resumen usando PyTorch
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Función para resumir un lote de textos
def summarize_batch(texts, min_length_ratio=0.2, max_length_ratio=0.5):
    global contador
    summaries = []

    for text in texts:
        input_length = len(text.split())

        # Calcular max_length y min_length ajustados
        max_length = min(int(input_length * max_length_ratio), input_length)
        min_length = min(int(input_length * min_length_ratio), input_length)

        # Asegurarse de que max_length no sea menor que min_length
        if max_length < min_length:
            max_length = min_length + 1

        # Generar resumen
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

        contador += 1
        print(f"Número: {contador}         Resumen: {summary[0]['summary_text']}")
        summaries.append(summary[0]['summary_text'])

    return summaries

# Procesamiento por lotes
batch_size = 1
resumenes = []

for i in range(0, len(df), batch_size):
    batch_texts = df['Text'].iloc[i:i + batch_size].tolist()
    batch_resumenes = summarize_batch(batch_texts)
    resumenes.extend(batch_resumenes)

# Asignar los resúmenes a la columna 'Resumen'
df['Resumen'] = resumenes

# Guardar el resultado en un archivo CSV
df.to_csv('Reviews_resumido.csv', index=False)
