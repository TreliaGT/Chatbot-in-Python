import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model
# Download necessary NLTK packages
nltk.download('punkt') 
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

# Load pre-trained words and classes
words = pickle.load(open('words.pkle', 'rb'))
classes = pickle.load(open('classes.pkle' , 'rb'))
model = load_model('chatbot_model.h5')

# Function to clean up sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Convert sentence to bag of words (feature vector)
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i ,word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predict the class of a sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes [r[0]], 'probability': str(r[1])})
    return return_list

# Get response from the intents based on predicted class
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print ('GO!, Bot is Running')

while True:
    message = input('')
    if message.lower() in ['exit', 'quit', 'bye']:
        print("Goodbye!")
        break
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)