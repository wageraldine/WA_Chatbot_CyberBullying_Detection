import json
import requests
import datetime
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class ultraChatBot():    
    def __init__(self, json):
        self.json = json
        self.dict_messages = json['data']
        self.ultraAPIUrl = 'https://api.ultramsg.com/instance68139/'
        self.token = 'eb2ht67nwafkujyu'
   
    def send_requests(self, type, data):
        url = f"{self.ultraAPIUrl}{type}?token={self.token}"
        headers = {'Content-type': 'application/json'}
        answer = requests.post(url, data=json.dumps(data), headers=headers)
        return answer.json()

    def sentiment_classification(self, chatID, text):        
        modelname = 'model_cyberbulying_classification.sav'
        loaded_model = pickle.load(open(modelname, 'rb'))
        vec_path = 'vectorizer_cyberbulying.pickle'
        tfidf_file = open(vec_path, 'rb')
        tfidfconverter = pickle.load(tfidf_file)
        tfidf_file.close()

        text_vector = tfidfconverter.transform([text]).toarray()
        pred_text = loaded_model.predict(text_vector)  
            
        pkl_file = open('encoder_cyberbulying.pkl', 'rb')
        le = pickle.load(pkl_file) 
        pkl_file.close()
            
        pred_text = le.inverse_transform(pred_text)
        score = round(max(loaded_model.predict_proba(text_vector)[0])*100,2)
        result = str(pred_text[0].capitalize())

        if result == "Negative":
            respons = "*Sebanyak " + str(score) + "% dari perkataan anda mengandung Ucapan Kasar !!*"
        else:
            respons = "*Perkataan anda tidak mengandung indikasi Cyber Bulying*"

        data = {"to" : chatID,
                "body" : respons}  
        answer = self.send_requests('messages/chat', data)

        return answer

    def Processingـincomingـmessages(self):
        if self.dict_messages != []:
            message =self.dict_messages
            text_list = message['body'].split()
            text = " ".join(text_list) 
            if not message['fromMe']:
                chatID  = message['from'] 
                self.sentiment_classification(chatID, text)
            else: return 'NoCommand'