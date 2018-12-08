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
frac = 15      #Factor to Downsample due to runtime constraints 
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



#########
### Baseline
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
# split the dataset into training and validation datasets 




'''count_vect = CountVectorizer(stop_words = 'english',analyzer='word', token_pattern=r'\w{1,}')
#fill in nan to make sklearn happy
count_vect.fit(train_df['text'])
# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)'''


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    classifier.fit(feature_vector_train, label)
    pred_train = classifier.predict(feature_vector_train)
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.classification_report(label, pred_train), metrics.classification_report(valid_y, predictions)
#confusion matrix on validation matrix
#metrics.confusion_matrix(label, pred_test), metrics.confusion_matrix(valid_y, predictions)

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train_df['text'], y, test_size = .2)
# Create CountVectorizer
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(train_df['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)
np.nan_to_num(xtrain_tfidf)
np.nan_to_num(xvalid_tfidf)


train, test = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)



#np.nan_to_num(xtrain_count)
#np.nan_to_num(xvalid_count)
#conf_test, conf_valid = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print(train)
print(test)
