from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas as pd, xgboost, numpy as np, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import os

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
frac = 10       #Factor to Downsample due to runtime constraints 
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
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(train_df['labels'])


####Cleaning 
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

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train_df['text'], y, test_size = .2)


#fill in nan to make sklearn happy
train_df['text'] = train_df['text'].fillna(' ')
train_x = train_x.fillna(' ')
valid_x = valid_x.fillna(' ') 
#ngram
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(train_df['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    classifier.fit(feature_vector_train, label)
    pred_train = classifier.predict(feature_vector_train)
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
        pred_train = pred_train.argmax(axis = -1)

    return metrics.classification_report(label, pred_train), metrics.classification_report(valid_y, predictions)

def create_model_architecture(input_size):
    # create input layer 
    input_layer = layers.Input((input_size, ), sparse=True)
    
    # create hidden layer
    hidden_layer1 = layers.Dense(100, activation='relu')(input_layer)
    hidden_layer2 = layers.Dense(20, activation="relu")(hidden_layer1)
    
    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer2)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier 

classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])
train,test = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, is_neural_net=True)


np.nan_to_num(xtrain_tfidf_ngram)
np.nan_to_num(xvalid_tfidf_ngram)
conf_test, conf_valid = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print(conf_test)
print(conf_valid)