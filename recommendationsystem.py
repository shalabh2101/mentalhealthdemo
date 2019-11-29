import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import random
import string
import copy  
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
import csv


def csv_update(resp_obj):
    with open('data/responses.csv', mode='a') as csv_file:
        fieldnames = ['name', 'lastName', 'Date', 'age']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        temp = {
            "name": resp_obj['name'],
            "lastName": resp_obj['lastName'],
            "Date": "asd",
            "age": 29
        }
        writer.writerow(temp)


data_location = 'data/responses.csv'
data = pd.read_csv(data_location)

app = Flask(__name__)

def encoding(data):
    df = copy.deepcopy(data)
    for i in df.select_dtypes(include=['object']).columns:
        list_unique = set(df[i].unique())
        dict_pro = dict(zip(list_unique,np.arange(len(list_unique))))
        df[i] = df[i].map(dict_pro)
    return df

data.select_dtypes(include=['object']).columns
data = encoding(data)

for i in range(data.shape[0]):
    data['title']=i
for i in range(data.shape[0]):
    data['title'][i]=str(i)+random.choice(string.ascii_letters)    

indices = pd.Series(data.index, index=data['title']).drop_duplicates()

indices = pd.Series(data.index, index=data['title']).drop_duplicates()
data = data.dropna()
data = data.dropna()
feature_vactor=data[['Music','Dance','Musical','Pop']]
f=data[['title','Music','Dance','Musical','Pop']]


cosine_sim = linear_kernel(feature_vactor, feature_vactor)
cosine_sim2 = cosine_similarity(feature_vactor, feature_vactor)

def get_recommendations(title, cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[movie_indices]

data = data.reset_index()
indices = pd.Series(data.index, index=data['title'])

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        resp = request.json
        # csv_update(resp)
        res = get_recommendations(resp['name'], cosine_sim2)
        print(res)
        res = res.values.tolist()
        response = jsonify({
            "result": res
        })
        return response
    else:
        return "get method!!!"


#mental health prediction
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import randint

# prep
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler

# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier



#age_df = pd.read_csv('./Data/mental-heath-in-tech-2016_20161114.csv')

train_df = pd.read_csv('./data/processed.csv')

# define X and y
feature_cols = ['What is your age?', 'What is your gender?', 'Do you have a family history of mental illness?',
                'Does your employer provide mental health benefits as part of healthcare coverage?',
                'Do you know the options for mental health care available under your employer-provided coverage?',
                'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?', 
                'If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:',
                'Do you believe your productivity is ever affected by a mental health issue?']
X = train_df[feature_cols]
y = train_df['Have you ever sought treatment for a mental health issue from a mental health professional?']

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)


def treeClassifier(c):
    # Calculating the best parameters
    tree = DecisionTreeClassifier()
    featuresSize = feature_cols.__len__()
    param_dist = {"max_depth": [3, None],
              "max_features": randint(1, featuresSize),
              "min_samples_split": randint(2, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}
    #tuningRandomizedSearchCV(tree, param_dist)
    
    # train a decision tree model on the training set
    tree = DecisionTreeClassifier(max_depth=3, min_samples_split=8, max_features=6, criterion='entropy', min_samples_leaf=7)
    tree.fit(X_train, y_train)
    
    # make class predictions for the testing set
   # y_pred_class = tree.predict(X_test)

   
    c=np.array(c).reshape(-1, 8)
 
    y_pred_class = tree.predict(c)
    print('y_pred_class')
    return (y_pred_class)

##all has been from Machine Learning for Mental Health
##Nueral Network Remaining
##Prediction model alpha values aka coefficients are remaining

@app.route('/health', methods=['GET','POST'])
def index2():
    if request.method == 'POST':
        resp = request.json
        # csv_update(resp)
        test_data=[resp['1'],resp['2'],resp['3'],resp['4'],resp['5'],resp['6'],resp['7'],resp['8']]
        res = treeClassifier(test_data)
        print(res)
        res = res.values.tolist()
        response = jsonify({
            "result": res
        })
        return response
    else:
        return "get method!!!"


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)