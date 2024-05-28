import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import nltk
import sqlite3

# Load NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load pre-trained model and data
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

# Connect to SQLite database
conn = sqlite3.connect('chat_history.db')
c = conn.cursor()

# Create table to store chat history
c.execute('''CREATE TABLE IF NOT EXISTS chat_history
             (user_message TEXT, bot_response TEXT)''')
conn.commit()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return np.array(bag)

def predict_class(sentence):
    p = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

class ChatApp(App):
    
    def build(self):
        self.layout = BoxLayout(orientation="vertical")

        # Change background color to blue
        self.layout.background_color = (0, 0, 1, 1)

        # Create ScrollView for chat history
        self.chat_history = ScrollView()
        self.chat_log = Label(size_hint_y=None)
        self.chat_history.add_widget(self.chat_log)
        self.layout.add_widget(self.chat_history)

        # Create TextInput for user input
        self.user_input = TextInput(size_hint_y=None, height=100)
        self.layout.add_widget(self.user_input)

        # Create Send Button
        self.send_button = Button(text="Send", size_hint_y=None, height=50)
        self.send_button.bind(on_press=self.send_message)
        self.layout.add_widget(self.send_button)

        # New chat option button
        self.new_chat_button = Button(text="Start New Chat", size_hint_y=None, height=50)
        self.new_chat_button.bind(on_press=self.start_new_chat)
        self.layout.add_widget(self.new_chat_button)

        # Load chat history from database
        self.load_chat_history()

        return self.layout

    def send_message(self, instance):
        msg = self.user_input.text.strip()
        self.user_input.text = ""

        if msg != '':
            # Append user message to chat history
            self.chat_log.text += f"You: {msg}\n\n"
            # Call functions to predict and respond
            ints = predict_class(msg)
            res = getResponse(ints, intents)
            # Append bot response to chat history
            self.chat_log.text += f"Cruella: {res}\n\n"
            # Store chat history in database
            self.store_chat_history(msg, res)
            # Scroll to the bottom of chat history
            self.chat_history.scroll_y = 0
            # Ensure the scroll view is scrolled to the bottom
            self.chat_history.scroll_to(self.chat_log)

    def start_new_chat(self, instance):
        # Clear chat history
        self.chat_log.text = ""
        # Clear chat history in database
        c.execute("DELETE FROM chat_history")
        conn.commit()

    def load_chat_history(self):
        # Load chat history from database
        c.execute("SELECT * FROM chat_history")
        rows = c.fetchall()
        for row in rows:
            self.chat_log.text += f"You: {row[0]}\n\n"
            self.chat_log.text += f"Bot: {row[1]}\n\n"

    def store_chat_history(self, user_msg, bot_response):
        # Store chat history in database
        c.execute("INSERT INTO chat_history (user_message, bot_response) VALUES (?, ?)", (user_msg, bot_response))
        conn.commit()

if __name__ == "__main__":
    ChatApp().run()
