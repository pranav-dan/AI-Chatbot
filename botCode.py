
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
from nltk.tokenize import word_tokenize

from tensorflow.keras.models import load_model

nltk.download("punkt")
nltk.download("wordnet")


f = open('chatbot/sampleData.json')
intents=json.load(f)
#print(intents)

lemmatizedWord=pickle.load(open('../Pranav/words.pkl','rb'))
lemmatizedClass=pickle.load(open('../Pranav/classes.pkl','rb'))
#model=load_model('../Pranav/chatbotmodel.model')
model=load_model('../Pranav/chatbotmodel.h5')

lemmatizer=WordNetLemmatizer()


punctuation=["!","?",",",".","@","#",";",":","'","*"]

def clean(sentence):
        sent=word_tokenize(sentence)
        lemmatized_Sent=[lemmatizer.lemmatize(w) for w in sent if w not in punctuation]
        #print(lemmatized_Sent)
        return lemmatized_Sent

def getBOW(sentence):
    tokkenWord=clean(sentence)
    #bow=[]
    bow=[0]*len(lemmatizedWord)
    # for w in lemmatizedWord:
    #     if w in tokkenWord:
    #         bow.append(1)
    #     else:
    #         bow.append(0)
    for w in tokkenWord:
        for i, word in enumerate(lemmatizedWord):
            if word==w:
                bow[i]=1

    #print(bow)     
    return np.array(bow)


def predictClass(sentence):
    bow=getBOW(sentence)
    res  = model.predict(np.array([bow]))[0]
    #print(res)
    error_threshold=0.25
    results=[[i,r] for i,r in enumerate(res) if r > error_threshold]
    #print(results)
    results.sort(key=lambda  x:x[1], reverse=True)
    return_list=[]
    for r in results:
        return_list.append({"intents":lemmatizedClass[r[0]] , 'probability':str(r[1])})
        #return_list.append(lemmatizedClass[r[0]])
    #print("return list is:\n")
    #print(return_list)
    return return_list

def getResponse(intents_list,intents_json):
    tag=intents_list[0]["intents"]
    #print(intents_json["data"])
    info=intents_json["data"]
    list_of_intent=info["intents"]
    for i in list_of_intent:
        if i["tag"]==tag:
            result=random.choice(i["responses"])
            break
    return result


print("Running")


while True:
    message=input("")
    ints=predictClass(message)
    res=getResponse(ints,intents)
    print(res)    