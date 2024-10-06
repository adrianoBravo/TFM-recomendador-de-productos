# Importar las librerías necesarias
from transformers import BertTokenizer, TFBertForSequenceClassification, DataCollatorWithPadding
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from sklearn.metrics import f1_score
import os
from tensorflow.keras.callbacks import TensorBoard

# 1. Cargar y preprocesar los datos
df = pd.read_csv('Reviews.csv')

# Eliminar filas con valores nulos en las columnas importantes
df = df.dropna(subset=['Summary', 'Text', 'Score', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time'])

# Limitar el conjunto de datos a 25,000 muestras para reducir el tiempo de entrenamiento
df = df.iloc[:25000]

# 2. Codificar las etiquetas (ID de productos) como números
label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['ProductId'])

num_labels = len(label_encoder.classes_)  # Número de clases (productos distintos)

# 3. Estandarizar las características numéricas
scaler = StandardScaler()
df['scaled_score'] = scaler.fit_transform(df[['Score']])
df['scaled_helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'].replace(0, 1)  # Evitar división por 0
df['scaled_time'] = scaler.fit_transform(df[['Time']])

# 4. Dividir el conjunto de datos en entrenamiento y prueba
train_df, test_df = train_test_split(df.loc[:, ["Summary", "Text", "labels", "scaled_score", "scaled_helpfulness", "scaled_time"]],
                                     test_size=0.2, random_state=42)

# Convertir los DataFrames de pandas en Datasets de Hugging Face
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# 5. Cargar el tokenizer y el modelo de BERT preentrenado
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# 6. Tokenización con las características adicionales (puntuación, utilidad, tiempo)
def tokenize_function(example):
    tokenized_input = tokenizer(example["Summary"], example["Text"], truncation=True, max_length=128)
    tokenized_input["scaled_score"] = example["scaled_score"]
    tokenized_input["scaled_helpfulness"] = example["scaled_helpfulness"]
    tokenized_input["scaled_time"] = example["scaled_time"]
    return tokenized_input

# Aplicar la tokenización a los datasets
tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

# 7. Crear datasets de entrenamiento y prueba compatibles con tf.data
batch_size = 20

train_dataset = tokenized_datasets['train'].to_tf_dataset(
    columns=['input_ids', 'attention_mask', 'scaled_score', 'scaled_helpfulness', 'scaled_time'],
    label_cols=["labels"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator
)

test_dataset = tokenized_datasets['test'].to_tf_dataset(
    columns=['input_ids', 'attention_mask', 'scaled_score', 'scaled_helpfulness', 'scaled_time'],
    label_cols=["labels"],
    batch_size=batch_size,
    collate_fn=data_collator
)

# 8. Modificar la arquitectura del modelo para incluir las características adicionales
input_ids = tf.keras.layers.Input(shape=(None,), name='input_ids', dtype='int32')
attention_mask = tf.keras.layers.Input(shape=(None,), name='attention_mask', dtype='int32')
scaled_score = tf.keras.layers.Input(shape=(1,), name='scaled_score')
scaled_helpfulness = tf.keras.layers.Input(shape=(1,), name='scaled_helpfulness')
scaled_time = tf.keras.layers.Input(shape=(1,), name='scaled_time')

# Salida de BERT
bert_output = model.bert(input_ids, attention_mask=attention_mask)[0]
pooled_output = bert_output[:, 0, :]  # Usar el primer token (CLS)

# Concatenar las características adicionales
concatenated_output = tf.keras.layers.Concatenate()([pooled_output, scaled_score, scaled_helpfulness, scaled_time])

# Capas adicionales
dense_output = tf.keras.layers.Dense(128, activation='relu')(concatenated_output)
dropout_output = tf.keras.layers.Dropout(0.1)(dense_output)  # Capa de dropout para evitar sobreajuste
final_output = tf.keras.layers.Dense(num_labels)(dropout_output)

# Crear el modelo final con las entradas y salidas modificadas
improved_model = tf.keras.Model(inputs=[input_ids, attention_mask, scaled_score, scaled_helpfulness, scaled_time], outputs=final_output)

# 9. Compilar el modelo con un scheduler de tasa de aprendizaje polinomial
num_epochs = 50
num_train_steps = len(train_dataset) * num_epochs
lr_scheduler = PolynomialDecay(
    initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_train_steps
)
opt = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

# Usar la pérdida de entropía cruzada categórica
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
improved_model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

# 10. Calcular el peso para cada clase
class_weights = {i: 1.0 / count for i, count in enumerate(df['labels'].value_counts())}
max_weight = max(class_weights.values())
class_weights = {i: max_weight / weight for i, weight in class_weights.items()}

# 11. Configurar TensorBoard para monitorear métricas durante el entrenamiento
log_dir = os.path.join("logs", "fit", "bert_product")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Entrenar el modelo
improved_model.fit(train_dataset, epochs=num_epochs, callbacks=[tensorboard_callback])

# Guardar el modelo entrenado
improved_model.save("models/bert-product-recommendation-mejorada-zero-shot-final")

# 12. Evaluar el modelo en el conjunto de prueba
true_labels = []
predictions = []

for batch in test_dataset:
    inputs, labels = batch
    logits = improved_model.predict({
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'scaled_score': inputs['scaled_score'],
        'scaled_helpfulness': inputs['scaled_helpfulness'],
        'scaled_time': inputs['scaled_time']
    }, verbose=0)
    preds = tf.argmax(logits, axis=-1)
    true_labels.extend(labels.numpy())
    predictions.extend(preds.numpy())

# 13. Calcular el F1 Score
true_labels = np.array(true_labels)
predictions = np.array(predictions)
f1 = f1_score(true_labels, predictions, average='weighted')
print(f"F1 Score: {f1}")

# Instrucción para visualizar las métricas en TensorBoard
print("Para visualizar el entrenamiento, ejecuta: tensorboard --logdir=logs/fit/bert_product")
