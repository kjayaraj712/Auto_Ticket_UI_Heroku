from flask import Flask, render_template, request, json, jsonify, redirect, url_for
import h5py
import sys
import re
import glob
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import helper
import pickle
import tensorflow_hub as hub
import tokenization
import html
from sklearn import preprocessing
import os
sys.path.append(os.path.abspath('./models'))

# load model
model = tensorflow.keras.models.load_model('./models')

# getting the pre-trained bert layer encodings and tokenize the data
bert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1', trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
le = preprocessing.LabelEncoder()

assignor = helper.Assignor()

enc = {
    0 : 'GRP_0',
    1: 'GRP_12',
    2: 'GRP_19',
    3: 'GRP_2',
    4: 'GRP_3',
    5: 'GRP_6',
    6: 'GRP_8',
    7: 'GRP_9',
    8: 'Others'
}

#app
app = Flask(__name__)

def bert_encode(texts, tokenizer, max_len=512):
    '''
    This function returns encodings based on the tokenizer passed as a parameter.
    '''
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])

def predict():
    # get data

    if request.method == 'POST':
        message = request.form['message']
        

        ## Clean and preprocess
        message = assignor.preprocess_input(np.expand_dims(message,axis=0))
        
        X_u_e = bert_encode(message, tokenizer, 400)
        
        # predictions
        result = model.predict(X_u_e)
        
        pred = np.argmax(model.predict(X_u_e),axis=1)
        

    # return data
    return render_template('result.html',prediction = enc[pred[0]])

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
