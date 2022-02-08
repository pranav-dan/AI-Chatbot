
##  Data loading and preprocessing

import json
import string
import random
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


nltk.download("punkt")
nltk.download("wordnet")
nltk.download('stopwords')
nltk.download('omw-1.4')

import tensorflow as tf 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout


#data=json.loads(open("sampledData".json.read()))
f = open('chatbot/sampleData.json')
data=json.load(f)


words=[]
classes=[]    ## contains tags 
patternDoc=[]
tagDoc=[]

lemmatizer=WordNetLemmatizer()
stop_words=stopwords.words('english')

data=data['data']


for info in data["intents"]:
    #print(info)
    for pattern in info[ "patterns"]:
        #print(pattern)
        token=nltk.word_tokenize(pattern)  
        words.extend(token)
        patternDoc.append(pattern)
        tagDoc.append(info["tag"])
    #tag=info["tag"]
    #tagDoc.append(tag)
    classes.append(info["tag"])
#print(tagDoc)
    
#print(words)
punctuation=["!","?",",",".","@","#",";",":","'","*"]

words=set(words)  # to remove the duplicate entries
classes=set(classes)  


      
## Now lemmatize the words from the list

lemmatizedWord=[]
lemmatizedClass=[]

for word in words:
    if word not in punctuation:
        lemmatizedWord.append(lemmatizer.lemmatize(word.lower()))

#print(lemmatizedWord)
for c in classes:
    if c not in punctuation:
        lemmatizedClass.append(lemmatizer.lemmatize((c.lower())))


pickle.dump(lemmatizedWord,open('words.pkl','wb'))
pickle.dump(lemmatizedClass,open('classes.pkl','wb'))

## converting text to numerics using bag of words
# training will contain the numeric data of both pattern and its respective tag


training=[]
initialOutput=[0]*len(classes)

for idx,pattern in enumerate(patternDoc):
    text=lemmatizer.lemmatize(pattern.lower())
    bagOfWords=[]
    for word in lemmatizedWord:
        if word in text:
            bagOfWords.append(1)
        else:
            bagOfWords.append(0)
        
    row=list(initialOutput)
    row[lemmatizedClass.index(tagDoc[idx])]=1
    training.append([bagOfWords,row])
print(training)
random.shuffle(training)
training=np.array(training)



## read this again
train_X = np.array(list(training[:, 0]))        ## contain the pattern in vector form
print(train_X)
train_y = np.array(list(training[:, 1]))        ## contain tags in vector form

#print(train_y)

## building neural network

input_nodes = (len(train_X[0]),)     ## check this if not working
output_nodes = len(train_y[0])
epochs = 200

# the deep learning model
model = Sequential()
model.add(Dense(128, input_shape=input_nodes, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(output_nodes, activation = "softmax"))

adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
#sgd=tf.keras.optimizers.SGD(lr=0.01,decay=1e-6,momentum=0.9,nestrerov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=["accuracy"])
#print(model.summary())
hist=model.fit(x=train_X, y=train_y, epochs=200, verbose=1, batch_size=5)
model.save('chatbotmodel.h5',hist)

#print("working")

