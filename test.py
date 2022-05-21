import numpy as np
import pickle
import random
import json
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('merged_dataset_intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return(np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    # print("Result is ---->", res)
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


# intents_Mental_Health_FAQ.json 0.4 confidence
# dialog_intents.json 0.3 confidence
# depression_chatbot_intents.json 0.6 confidence
# chatbot_chitchat_intents.json 0.65 confidence

def chatbot_response(msg):
    ints = predict_class(msg, model)
    # 79% confidence
    print(msg, ints)
    try:
        if(float(ints[0]['probability']) > 0.79):  # 79
            print("in accept by probability ")
            res = getResponse(ints, intents)
        elif(len(ints) == 1):
            print("in accept by scarcity ")
            res = getResponse(ints, intents)
        else:
            print("in reject ")
            res = "Umm, sorry I coundn't process it. Would you mind changing the words a bit so that I can interpret it correctly. I am not that smart you see."
    except:
        print("Exception")
        res = "Umm, sorry I coundn't process it. Would you mind changing the words a bit so that I can interpret it correctly. I am not that smart you see."
    return res
