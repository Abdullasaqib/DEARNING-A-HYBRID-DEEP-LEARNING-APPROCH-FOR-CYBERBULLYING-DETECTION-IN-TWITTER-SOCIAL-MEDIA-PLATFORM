import pandas as pd
import numpy as np
import pickle
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, SpatialDropout1D

print("Starting fast training...", flush=True)

try:
    df = pd.read_csv('cyberbullying_tweets.csv')
    print("Data loaded.", flush=True)
except Exception as e:
    print(f"Error loading data: {e}", flush=True)
    exit()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['cleaned_text'] = df['tweet_text'].apply(clean_text)
le = LabelEncoder()
df['label_id'] = le.fit_transform(df['cyberbullying_type'])
class_names = le.classes_

with open('class_names.pickle', 'wb') as f:
    pickle.dump(class_names, f)

MAX_NB_WORDS = 5000 # Reduced
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 50 # Reduced

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(df['cleaned_text'].values[:2000]) # Fit on subset
X = tokenizer.texts_to_sequences(df['cleaned_text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
Y = pd.get_dummies(df['cyberbullying_type']).values

with open('tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)

# TINY DATASET FOR SPEED
X_train = X[:500]
Y_train = Y[:500]

print("Building model...", flush=True)
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(class_names), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training...", flush=True)
model.fit(X_train, Y_train, epochs=1, batch_size=16, verbose=1)

print("Saving...", flush=True)
model.save('cyberbullying_model.h5')
print("DONE.", flush=True)
