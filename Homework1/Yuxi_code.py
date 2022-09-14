import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import style
style.use('ggplot')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
plt.rcParams['font.sans-serif'] = ['Simhei']

import re
from collections import Counter
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import numpy as np
import math
from math import sqrt

# ---------------------------- My k-NN implementation
class kNNClassifier:
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        return self

    def predict(self, x_predict):
        y_predict = [self._predict(x) for x in x_predict]
        return np.array(y_predict)

    def _predict(self, x):
        a = np.asarray(self)
        print(self.x_train)
        print(self.x_train.toarray())
        print(x)
        distances = []
        for x_train in self.x_train:
            distance = 0
            x_diff = x_train - x
            x_array = x_diff.toarray()
            for value in x_array[0]:
                distance += pow(value, 2)
            distance = math.sqrt(distance)
            distances.append(distance)
        nearest = np.argsort(distances)[:self.k]
        top_k_y = [self.y_train[index] for index in nearest]
        d = {}
        print(top_k_y)
        for cls in top_k_y:
            for num in cls:
                print(d.get(num, 0))
                d[num] = d.get(num, 0) + 1
        d_list = list(d.items())
        d_list.sort(key=lambda x: x[1], reverse=True)
        return np.array(d_list[0][0])

    def __repr__(self):
        return "KNN(k={})".format(self.k)


# --------------------------------------- Read file and set panda data type
data = pd.read_csv(r'./484_train_file.txt', sep="\t", header=0)
data

test_file = './484_test_file.txt'
lines = []
with open(test_file, 'r',encoding='utf-8') as file_to_read:
    while True:
        line = file_to_read.readline()
        if not line:
            break
        line = line.strip('\n')
        lines.append(line)
len(lines)

test_data = pd.DataFrame(lines, columns=['review'])
test_data


# ------------------------------------ Get the list of the sentiment of train data
data_sentiment= data.sentiment
data_sentiment
data.head()
data['sentiment'].value_counts()

Y_train = []
for sentiment in data_sentiment:
    y_sentiment= [str(sentiment)]
    Y_train.append(y_sentiment)
Y_train



# -----------------------Get the list of the review of train dataset and test dataset
review = data['review']
for i in range(0,5):
    print(review[i])
    print('---------------------train')
test_review = test_data['review']
for i in range(0,5):
    print(test_review[i])
    print('---------------------test')


# -------------- Replace the number, html tag and punctuation
# -------------- in the review of train dataset and test dataset
info = re.compile('0|1|2|3|4|5|6|7|8|9|0|<br />|[^\w\s]|_')
review = review.apply(lambda x: info.sub('', x))  # 替换所有匹配项
test_review = test_review.apply(lambda x: info.sub('', x))

for i in range(0, 5):
    print(review[i])
    print('-----------')

for i in range(0, 5):
    print(test_review[i])
    print('-----------')


# ------------------ Data Preprocessing: tokenize, remove stopwords and make stemming

# ############################ Data preprocessing for Train dataset
# tokenize
review_tokenize=review[:]
i=0
for sentence in review:
    review_tokenize[i] = word_tokenize(sentence)
    i+=1
print('Train after tokenize')
print(review_tokenize)
print('Train[0] after tokenize')
print(review_tokenize[0])
print('\n')


# delete stopwords
stwords=stopwords.words('english')
i=0
for sentence in review_tokenize:
    word_to_delete={}
    for word in sentence:
        if word.lower() in stwords:
            word_to_delete[word] = word_to_delete.get(word,0) + 1
    for key in word_to_delete:
        j=0
        while j < word_to_delete[key]:
            review_tokenize[i].remove(key)
            j+=1
    i+=1
print('Train after tokenize and stopwords')
print(review_tokenize)
print('Train[0] after tokenize and stopwords')
print(review_tokenize[0])
print('\n')


# Stemming
porter_stemmer = PorterStemmer()
i=0
for sentence in review_tokenize:
    review_tokenize[i] = [porter_stemmer.stem(word) for word in sentence]
    i+=1
print('Train after tokenize, stopwords and stemming')
print(review_tokenize)
print('Train[0] after tokenize, stopwords and stemming')
print(review_tokenize[0])
print('\n')

# ########################### Data preprocessing for Test dataset
# tokenize
test_review_tokenize=test_review[:]
i=0
for sentence in test_review:
    test_review_tokenize[i] = word_tokenize(sentence)
    i+=1
print('Test after tokenize')
print(test_review_tokenize)
print('Test[0] after tokenize')
print(test_review_tokenize[0])


# delete stopwords
stwords=stopwords.words('english')
i=0
for sentence in test_review_tokenize:
    word_to_delete={}
    for word in sentence:
        if word.lower() in stwords:
            word_to_delete[word] = word_to_delete.get(word,0) + 1
    for key in word_to_delete:
        j=0
        while j < word_to_delete[key]:
            test_review_tokenize[i].remove(key)
            j+=1
    i+=1
print('Test after tokenize and stopwords')
print(test_review_tokenize)
print('Test[0] after tokenize and stopwords')
print(test_review_tokenize[0])
print('\n')
print('word_to_delete[word]:')
print(word_to_delete)
print('\n')


# Stemming
porter_stemmer = PorterStemmer()
i=0
for sentence in test_review_tokenize:
    test_review_tokenize[i] = [porter_stemmer.stem(word) for word in sentence]
    i+=1
print('Test after tokenize, stopwords and stemming')
print(test_review_tokenize)
print('Test[0] after tokenize, stopwords and stemming')
print(test_review_tokenize[0])
print('\n')

# ----------------- Create lists for the review of train dataset and test dataset
# ----------------- after data preprocessing
X_train = []
X_test = []
for sentence in review_tokenize:
    string = ""
    first = 0
    for word in sentence:
        if first == 0:
            string += word
            first = 1
        else:
            string = string + " " + word
    X_train.append(string)
X_train

for sentence in test_review_tokenize:
    string = ""
    first = 0
    for word in sentence:
        if first == 0:
            string += word
            first = 1
        else:
            string = string + " " +word
    X_test.append(string)
X_test


# -------------- vectorize the review of the train dataset and test dataset
# -------------- dimensionality reduction conversion on word vectors
# -------------- let k-NN learn the train dataset and predict the test dataset
# -------------- the predict result will output to 'predict_text.txt'

# training set, test set division
# x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,test_size=0.2,random_state=30)


# Vectorize the review of the train dataset and test dataset
vector = CountVectorizer(binary=True, max_features=100)


# Dimensionality reduction conversion on word vectors
x_train_transform = vector.fit_transform(X_train)
x_test_transform = vector.transform(X_test)
y_train = Y_train
print("x_train_transform")
print(x_train_transform)
print("vocabulary")
print(vector.vocabulary_)
df1 = pd.DataFrame(x_train_transform.toarray(), columns=vector.get_feature_names())  # to DataFrame
df1
x_train_transform.shape
x_test_transform.shape


# let k-NN learn the train dataset and predict the test dataset
kNN_classify = kNNClassifier(17)
kNN_classify.fit(x_train_transform, y_train)
print("x_test")
print(x_test_transform)
y_predict = kNN_classify.predict(x_test_transform)
y_predict
len(y_predict)
# accuracy_score(y_test, y_predict)
y_predict = pd.DataFrame(y_predict)
y_predict.to_csv('predict_text.txt', index=False, header=None)