import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re
import os

# Load Resources
MODEL_PATH = "cyberbullying_model.h5"
TOKENIZER_PATH = "tokenizer.pickle"
CLASS_NAMES_PATH = "class_names.pickle"

print("Loading resources...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
with open(CLASS_NAMES_PATH, 'rb') as f:
    class_names = pickle.load(f)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

text = "hi"
cleaned = clean_text(text)
seq = tokenizer.texts_to_sequences([cleaned])
padded = pad_sequences(seq, maxlen=100)

print(f"Input: '{text}'")
print(f"Cleaned: '{cleaned}'")
print(f"Sequence: {seq}")
print(f"Padded shape: {padded.shape}")

pred = model.predict(padded)
print(f"Raw Prediction: {pred}")
label_id = np.argmax(pred, axis=1)[0]
confidence = float(np.max(pred))
predicted_class = class_names[label_id]

print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence}")

# Check if 'hi' is in vocabulary
if 'hi' in tokenizer.word_index:
    print(f"'hi' index: {tokenizer.word_index['hi']}")
else:
    print("'hi' is NOT in the vocabulary")
