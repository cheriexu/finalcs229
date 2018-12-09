from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas as pd, xgboost, numpy as np, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from nltk.stem.snowball import SnowballStemmer
# Others
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
from sklearn.manifold import TSNE


real_users = pd.read_csv('/home/cheriexu/genuine_accounts/users.csv',low_memory = False)
real_tweets = pd.read_csv('/home/cheriexu/genuine_accounts/tweets.csv',low_memory = False)
real_df = real_tweets.merge(real_users, on='id',how = 'outer', suffixes = ('_tweets','_users')) #Merging the user and tweets
real_df['labels']=pd.Series(np.ones(len(real_df['id']))) #Creating a new column to indicate that this is human data
#Get Bot data 
directory = os.fsencode('/home/cheriexu/russian-troll')
troll_data = pd.DataFrame()
for file in os.listdir(directory):
  print(file)
  filename = os.fsdecode(file)
  if filename.endswith(".csv"):
    botdata = pd.read_csv('/home/cheriexu/russian-troll/'+filename,low_memory = False)
    troll_data = troll_data.append(botdata) #append together
troll_data = troll_data.rename(index = str, columns = {"publish_date":"date"})
frac = 15       #Factor to Downsample due to runtime constraints 
troll_data['labels']=0
#Limiting too only English language tweets
english_bot = troll_data[troll_data['language']  == 'English']
troll_data['language'].unique()
print(len(english_bot),len(troll_data)) #Class imbalance though, we need more spam bot results
train_df = pd.DataFrame() #Create train data frame for splitting later
real_df = real_df.rename(index = str, columns = {"timestamp_tweets": "date"})
real= real_df[['text','labels','date']].dropna()
train_df = train_df.append(real[['text','labels','date']].sample(int(len(real_df)/frac)))
english_bot = english_bot.rename(index=str, columns={"content": "text"})
e_b = english_bot[['text','labels','date']].dropna()
train_df = train_df.append(e_b[['text','labels','date']].sample(int(len(english_bot)/frac)))
len(train_df)



nltk.download('stopwords')
def clean_text(text):
    ## Remove puncuation
    text = str(text)
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Stemming
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text

train_df['text'] = train_df['text'].map(lambda x: clean_text(x))
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(train_df['labels'])
vocabulary_size = 200000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(train_df['text'])
sequences = tokenizer.texts_to_sequences(train_df['text'])
data = pad_sequences(sequences, maxlen=50)
embeddings_index = dict()
f = open('/home/cheriexu/glove.twitter.27B.200d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
embedding_matrix = np.zeros((vocabulary_size, 200))
for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
model_glove = Sequential()
model_glove.add(Embedding(vocabulary_size, 200, input_length=50, weights=[embedding_matrix], trainable=False))
#model_glove.add(Dropout(0.2))
#model_glove.add(Conv1D(64, 5, activation='relu'))
#model_glove.add(MaxPooling1D(pool_size=4))
model_glove.add(LSTM(200,dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
model_glove.add(LSTM(200,dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
model_glove.add(Flatten())
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_glove.summary()

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train_df['text'], y, test_size = .2)
# Create CountVectorizer
sequences = tokenizer.texts_to_sequences(train_x)
data = pad_sequences(sequences, maxlen=50)
model_glove.fit(data, train_y, validation_split=0.1, epochs = 3)
sequences = tokenizer.texts_to_sequences(valid_x)
data = pad_sequences(sequences, maxlen=50)
pred = model_glove.predict_classes(data)
confusion_matrix = metrics.classification_report(valid_y,pred)

print(confusion_matrix)


#Normal conv_model

