from flask import Flask,jsonify,request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

app=Flask(__name__)
Swagger(app)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
reconstructed_model = keras.models.load_model("model")
MAX_SEQUENCE_LENGTH = 250
labels =['Animals', 'Compliment', 'Education', 'Health', 'Heavy Emotion', 'Joke',
       'Love', 'Politics', 'Religion', 'Science', 'Self'] 

@app.route('/',methods=["POST"])
def predict_note_authentication():
    """Let's identify the Context 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: Statement
        in: query
        type: string
        required: true
      
    responses:
        200:
            description: The output values
    """
    text=request.json['Statement']
    new_complaint=[text]
    seq = tokenizer.texts_to_sequences(new_complaint)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = reconstructed_model.predict(padded)

    return jsonify( [{'Statement':text,
                     'confidence_score':float(round(max(max(pred))*100,2)),
                     'predicted_label':labels[np.argmax(pred)]}])
if __name__=='__main__':
    app.run(debug=True)