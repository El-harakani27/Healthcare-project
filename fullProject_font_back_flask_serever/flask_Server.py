from flask import Flask,request,render_template
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
print(tf.__version__)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

##symp diagnoses model

all_symp = ['Acne', 'Arthritis', 'Bronchial Asthma', 'Cervical spondylosis',
       'Chicken pox', 'Common Cold', 'Dengue', 'Dimorphic Hemorrhoids',
       'Fungal infection', 'Hypertension', 'Impetigo', 'Jaundice',
       'Malaria', 'Migraine', 'Pneumonia', 'Psoriasis', 'Typhoid',
       'Varicose Veins', 'allergy', 'diabetes', 'drug reaction',
       'gastroesophageal reflux disease', 'peptic ulcer disease',
       'urinary tract infection']

try:
    with open('mohamed_tokenizer.pickle', 'rb') as t:
        tokenizer = pickle.load(t)
except Exception as e:
    print("Error loading tokenizer:", e)
model_symp = tf.keras.models.load_model('mohamed_symptom.h5')
def input_tokenizer(txt,tok):
  val=tok.texts_to_sequences(txt)
  p =pad_sequences(val,maxlen=43,padding='post',truncating='post')
  return p
def prediction_symptom(txt,classes,model):
  output=model.predict(txt)
  pred = np.argmax(output,axis=1)
  output_pred=classes[pred[0]]
  return output_pred

############################################################################################################################################
##ChatBot model
# loading the files we made previously
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model_chatbot = tf.keras.models.load_model('chatbotmodel.h5')

def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower())
                      for word in sentence_words]
    return sentence_words
def bagw(sentence):

	# separate out words from the input sentence
	sentence_words = clean_up_sentences(sentence)
	bag = [0]*len(words)
	for w in sentence_words:
		for i, word in enumerate(words):

			# check whether the word
			# is present in the input as well
			if word == w:

				# as the list of words
				# created earlier.
				bag[i] = 1
	# return a numpy array
	return np.array(bag)
def predict_class(sentence,classes):
    bow = bagw(sentence)
    res = model_chatbot.predict(np.array([bow]))
    cl =np.argmax(res,axis=1)
    p = classes[cl[0]]
    return p

def get_response(txt,classes,intents):
    p =predict_class(txt,classes)

  
    for i in intents['intents']:
        if p==i['tag']:
            respond =i['responses'] 
        
    return (respond)

############################################################################################################################################

##classification model
with open('model_heart.pkl','rb') as m:
    model =pickle.load(m)

##############################################################################################################################################
##detection
all_detect=['Covid', 'Normal', 'Viral Pneumonia']
model_detection = tf.keras.models.load_model('detection_model.h5')
def predict_transfer_detect(img,classes,model):
  r=tf.keras.layers.Resizing(256,256)
  read_im=plt.imread(img)
  out=r(read_im)
  p=np.argmax(model.predict(np.expand_dims(out,axis=0)),axis=1)
  val=classes[p[0]]
  return val 
##############################################################################################################################################
app = Flask(__name__)

@app.route('/flask', methods=['GET', 'POST'])
def index():
    dataa=request.get_json()
    vals = json.loads(dataa)
    x=np.array([]).astype(np.float16)
    v=dict(vals)
    for k,c in v.items():    
        x=np.append(x,c)

    x = np.expand_dims(x,axis=0)
    output=model.predict(x)
    s="Can't predict"
    if output[0]==1:
        s="You have heart diease"
    if output[0]==0:
        s="You don't have heart diease"
    return s

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/get',methods=['GET', 'POST'])
def get_respon():
     sentence = request.get_json()
     txt = json.loads(sentence)
     txt=txt['msg']
     print(txt)
     
     res = get_response(txt,classes,intents)
     return str(res[0]) 
@app.route('/symp',methods=['GET', 'POST']) 
def get_symp():
    long_text = request.get_json()
    my_text = json.loads(long_text)
    processed_text = input_tokenizer(my_text['symp'], tokenizer)
    
    # Get the diagnosis
    diagnosis = prediction_symptom(processed_text, all_symp, model_symp)
    return diagnosis
@app.route('/corona',methods=['GET', 'POST'])
def get_corona():
    s=''
    direct='F:\KnowledgeBaseProject\images'
    if os.listdir(direct)==[]:
        s='No Images Found'
    else:
        im_path = os.path.join(direct,os.listdir(direct)[-1])
        print(os.listdir(direct)[-1])
        val=predict_transfer_detect(im_path,all_detect,model_detection)
        print(val)
        s=val
    return s
if __name__ == "__main__":
    app.run(port=5000, debug=True)
