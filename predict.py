import sys
import json
import pickle
import string
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# NLTK has been removed to make the script simpler

def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'chatbot_model.h5')
        tokenizer_path = os.path.join(script_dir, 'tokenizer.pickle')
        label_encoder_path = os.path.join(script_dir, 'label_encoder.pickle')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing required file: {model_path}")

        model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open(label_encoder_path, 'rb') as enc_file:
            label_encoder = pickle.load(enc_file)

        max_len = 120

        def clean_text(text):
            text = text.lower()
            text = ''.join([char for char in text if char not in string.punctuation])
            tokens = text.split()
            # Lemmatization is removed
            return ' '.join(tokens)

        def predict_category(sentence):
            cleaned_sentence = clean_text(sentence)
            sequence = tokenizer.texts_to_sequences([cleaned_sentence])
            padded_sequence = pad_sequences(sequence, maxlen=max_len, truncating='post')
            prediction = model.predict(padded_sequence, verbose=0)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_category = label_encoder.inverse_transform([predicted_class_index])[0]
            return predicted_category

        if len(sys.argv) > 1:
            input_sentence = sys.argv[1]
            prediction = predict_category(input_sentence)
            print(json.dumps({"prediction": prediction}))
        else:
            print(json.dumps({"error": "No input sentence provided."}))

    except Exception as e:
        print(json.dumps({"error": f"An unexpected error occurred in predict.py: {str(e)}"}))

if __name__ == "__main__":
    main()