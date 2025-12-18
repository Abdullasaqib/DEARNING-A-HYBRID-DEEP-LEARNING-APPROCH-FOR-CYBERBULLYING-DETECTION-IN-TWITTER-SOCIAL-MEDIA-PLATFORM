import pandas as pd
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, SpatialDropout1D

# 1. Load Data
print("Loading data...")
try:
    df = pd.read_csv('cyberbullying_tweets.csv')
except FileNotFoundError:
    print("Error: 'cyberbullying_tweets.csv' not found.")
    exit()

# 2. Preprocess Data
print("Preprocessing data...")
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text) # Remove URLs
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

df['cleaned_text'] = df['tweet_text'].apply(clean_text)

# Encode Labels
le = LabelEncoder()
df['label_id'] = le.fit_transform(df['cyberbullying_type'])
class_names = le.classes_
print("Classes:", class_names)

# Save Label Encoder classes
with open('class_names.pickle', 'wb') as f:
    pickle.dump(class_names, f)

# Tokenization
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['cleaned_text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(df['cleaned_text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
Y = pd.get_dummies(df['cyberbullying_type']).values

# Save Tokenizer
with open('tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)

# Train/Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

# 3. Build Hybrid Model
print("Building model...")
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(class_names), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# 4. Train Model
print("Training model...")
epochs = 5 # Increased for better accuracy
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[])

# 5. Evaluate
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(score[0], score[1]))

# 6. Save Model
print("Saving model...")
model.save('cyberbullying_model.h5')
print("Model saved as 'cyberbullying_model.h5'")
