import json
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================
# 1. LOAD FILES
# ==============================

model = tf.keras.models.load_model("model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("datasets/navigation.json") as file:
    data = json.load(file)

# Get max length
max_len = model.input_shape[1]

# ==============================
# 2. PREDICTION FUNCTION
# ==============================

def predict_intent(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding="post")

    prediction = model.predict(padded, verbose=0)
    index = np.argmax(prediction)
    confidence = np.max(prediction)

    tag = label_encoder.inverse_transform([index])[0]
    return tag, confidence

# ==============================
# 3. CHAT LOOP
# ==============================

print("LICET Campus Navigation Bot Ready! (type 'exit' to quit)")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    tag, confidence = predict_intent(user_input)

    if confidence < 0.6:
        print("Bot: I'm not sure. Please ask clearly about campus locations.")
        continue

    for intent in data["intents"]:
        if intent["tag"] == tag:
            print("Bot:", random.choice(intent["responses"]))
            break