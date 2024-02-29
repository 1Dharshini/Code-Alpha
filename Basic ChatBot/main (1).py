import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

try:
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    print("Model loading...")
    model = load_model('chatterbot.h5')
    print("Model loaded successfully!")
except Exception as e:
    print("Error:", e)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for i, w in enumerate(words):
        if w in sentence_words:
            bag[i] = 1
    return np.array(bag)

def predict_class(sentence,model):
    try:
        bow = bag_of_words(sentence)
        result = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        result = [[i, r] for i, r in enumerate(result) if r > ERROR_THRESHOLD]
        result.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in result:
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        return return_list
    except Exception as e:
        print("error",e)
        return []

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that."

print("GO! BOT IS RUNNING!")

while True:
    try:
        message = input("")
        response = get_response(predict_class(message,model), intents)
        print(response)
    except Exception as e:
        print("error",e)
