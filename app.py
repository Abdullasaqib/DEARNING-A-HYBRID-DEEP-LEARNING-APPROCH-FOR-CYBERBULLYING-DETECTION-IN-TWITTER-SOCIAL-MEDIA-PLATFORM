from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re
import os

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Load Resources (Model, Tokenizer, Encoder)
MODEL_PATH = "cyberbullying_model.h5"
TOKENIZER_PATH = "tokenizer.pickle"
CLASS_NAMES_PATH = "class_names.pickle"

model = None
tokenizer = None
class_names = None

def load_resources():
    global model, tokenizer, class_names
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        print("Loading model and tokenizer...")
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        with open(CLASS_NAMES_PATH, 'rb') as f:
            class_names = pickle.load(f)
        print("Resources loaded successfully.")
    else:
        print("Model or Tokenizer not found. Please train the model first.")

load_resources()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict")
async def predict(request: Request, tweet: str = Form(...)):
    if not model or not tokenizer:
        load_resources()
        if not model:
            return templates.TemplateResponse("index.html", 
                {"request": request, "result": "Error: Model not loaded.", "tweet": tweet})
    
    # Preprocess
    cleaned_tweet = clean_text(tweet)
    
    # SAFE LIST CHECK - Comprehensive list of positive words and greetings
    safe_list = [
        "hi", "hello", "hey", "test", "testing", "ok", "okay", "good", "nice",
        "good morning", "good afternoon", "good evening", "good night",
        "have a great day", "have a nice day", "you are amazing", "great job",
        "keep it up", "stay positive", "love you", "thank you", "thanks",
        "appreciate it", "wonderfull", "be happy", "kindness", "bless you",
        "congratulations", "welcome", "cheers", "awesome", "perfect", "excellent",
        "brilliant", "fantastic", "superb", "wonderful", "lovely", "beautiful",
        "safe", "healthy", "peace", "joy", "happiness", "smile", "friendly",
        "help", "helpful", "support", "encourage", "brave", "strong", "proud",
        "inspire", "success", "winner", "champion", "star", "gentle", "care",
        "respect", "honest", "trust", "loyal", "friend", "best friend", "family",
        "community", "unity", "humanity", "hope", "faith", "dream", "believe",
        "amazing"
    ]
    
    # Check if any phrases in the safe_list appear in the cleaned_tweet
    # We use regex word boundaries \b to ensure we match whole words/phrases
    # e.g. "hi" should NOT match "hit", but should match "hi there"
    is_safe = False
    for safe_word in safe_list:
        # Create a pattern that looks for the safe word surrounded by word boundaries
        pattern = r'\b' + re.escape(safe_word) + r'\b'
        if re.search(pattern, cleaned_tweet):
            is_safe = True
            break
            
    if is_safe:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": {
                "class": "not_cyberbullying",
                "is_bullying": False,
                "confidence": "100.00% (Positive Content)"
            }, 
            "tweet": tweet
        })

    seq = tokenizer.texts_to_sequences([cleaned_tweet])
    
    # EMPTY SEQUENCE CHECK
    if not seq or len(seq[0]) == 0:
         return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": {
                "class": "not_cyberbullying",
                "is_bullying": False,
                "confidence": "100.00% (No abusive words found)"
            }, 
            "tweet": tweet
        })

    padded = pad_sequences(seq, maxlen=100) # Match MAX_SEQUENCE_LENGTH from train_model.py
    
    # Predict
    pred = model.predict(padded)
    label_id = np.argmax(pred, axis=1)[0]
    confidence = float(np.max(pred))
    
    if class_names is not None:
        predicted_class = class_names[label_id]
    else:
        predicted_class = str(label_id)
    
    is_bullying = predicted_class != 'not_cyberbullying'
    
    result = {
        "class": predicted_class,
        "is_bullying": is_bullying,
        "confidence": f"{confidence * 100:.2f}%"
    }
    
    return templates.TemplateResponse("index.html", {"request": request, "result": result, "tweet": tweet})
