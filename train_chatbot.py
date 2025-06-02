import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import string

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout # type: ignore

# --- Automated NLTK Resource Download (for Lemmatizer) ---
# This section has the corrected exception handling.
try:
    nltk.data.find('corpora/wordnet')
except LookupError: # This is the corrected line.
    print("Downloading NLTK 'wordnet' model...")
    nltk.download('wordnet')
    print("'wordnet' model downloaded.")
print("NLTK resources are up to date.")
# ----------------------------------------

# --- 1. Data Loading and Preprocessing ---
lemmatizer = WordNetLemmatizer()

print("Step 1: Loading data...")
df = pd.read_csv('customer_support_tickets.csv')
df['text'] = df['Ticket Subject'] + ' ' + df['Ticket Description']
features = df['text']
labels = df['Ticket Type']
print("Data loaded and combined.")


# --- 2. Text Cleaning and Lemmatization ---
print("\nStep 2: Cleaning and preprocessing text...")
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split() # Using the reliable split() function
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

features = features.apply(clean_text)
print("Text cleaning complete.")


# --- 3. Encoding Labels and Tokenizing Text ---
print("\nStep 3: Encoding labels and tokenizing text...")
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

vocab_size = 10000
embedding_dim = 16
max_len = 120

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(features)

sequences = tokenizer.texts_to_sequences(features)
padded_sequences = pad_sequences(sequences, maxlen=max_len, truncating='post')
print("Label encoding and text tokenization complete.")


# --- 4. Splitting the Data ---
print("\nStep 4: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences,
    encoded_labels,
    test_size=0.2,
    random_state=42,
    stratify=encoded_labels
)
print("Data splitting complete.")


# --- 5. Building the Neural Network Model ---
print("\nStep 5: Building the model...")
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(24, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax') # Corrected a typo here too
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
print("Model built successfully.")


# --- 6. Training the Model ---
print("\nStep 6: Training the model...")
epochs = 30
history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    validation_data=(X_test, y_test),
    verbose=2
)
print("Model training complete.")


# --- 7. Saving the Model and Artifacts ---
print("\nStep 7: Saving model and necessary artifacts...")
model.save('chatbot_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(label_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

print("All files saved successfully! We are finally ready for the next phase!")