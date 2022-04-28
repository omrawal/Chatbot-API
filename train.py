import random
from keras.layers import Dense, Activation, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from keras.models import Sequential
import numpy as np
import pickle
import json
from tensorflow.keras.optimizers import SGD
import nltk
from nltk.stem import WordNetLemmatizer
from keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor='loss', min_delta=0,
                          patience=150, verbose=1, restore_best_weights=True)

lemmatizer = WordNetLemmatizer()


words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open("merged_dataset_intents.json").read()
intents = json.loads(data_file)
nltk.download('wordnet')

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower())
         for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print(len(documents), "documents")
# classes = intents
print(len(classes), "classes", classes)
# words = all words, vocabulary
print(len(words), "unique lemmatized words", words)


pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(
        word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model2 = Sequential()
model2.add(Dense(256,  activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(128,  activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model2.compile(loss='categorical_crossentropy',
               optimizer=sgd, metrics=['accuracy'])
X_train = np.array(train_x)
y_train = np.array(train_y)
# Further training model 2
hist3 = model2.fit(X_train, y_train,
                   epochs=1500, batch_size=64, verbose=1,
                   callbacks=[earlystop]
                   )

# Model has 93.22% accuracy at patience = 150 loss 19.78
# Saving accuracy
model2.save('chatbot_model.h5', hist3)
print("model created")
