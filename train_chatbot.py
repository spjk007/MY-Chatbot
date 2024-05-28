import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

# Load intents from the new intents file
intents_file = open('intents.json').read()
intents = json.loads(intents_file)

# Initialize variables
words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']
lemmatizer = WordNetLemmatizer()

# Process intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        # Add documents to the corpus
        documents.append((word, intent['tag']))
        # Add to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lowercase, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Create bag of words for each pattern
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# Shuffle and convert to numpy array
random.shuffle(training)
bags_of_words = np.array([x[0] for x in training])
output_rows = np.array([x[1] for x in training])

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(bags_of_words.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_rows.shape[1], activation='softmax'))

# Compile model
sgd = SGD(momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
history = model.fit(bags_of_words, output_rows, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5')

print("Model created and saved.")
