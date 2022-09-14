import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("The program is running, please wait for a while...")

# read the '.csv' data
data = pd.read_csv('./train_A.csv', header = None)
predict_features = pd.read_csv('./test_A.csv', header = None)

# get the label 0 and label 1 data
data_0 = data[data[10]==0]
data_1 = data[data[10]==1]

# there are 1909 label 0 data and only 540 label 1 data
# oversampling the label 1 data (540 -> 1080 -> 2160 -> 1909)
# make the label 0 and 1 data balanced
oversampling_data_1 = np.append(data_1, data_1, axis=0)
oversampling_data_1 = np.append(oversampling_data_1, oversampling_data_1, axis=0)
oversampling_data_1 = oversampling_data_1[:1909]
oversampling_data = np.append(data_0, oversampling_data_1, axis=0)
oversampling_data = pd.DataFrame(oversampling_data)

# shuffle
oversampling_data = shuffle(oversampling_data)

# renew the number of row
oversampling_data = oversampling_data.to_numpy()
oversampling_data = pd.DataFrame(oversampling_data)

# get the label of oversampling data
data_label = oversampling_data[10]
data_label = data_label.astype(int)

# split the dataset to validation set(3500) and test set(318)
# get the label and features part of validation set(3500) and test set(318)
data_label_3500 = data_label[:3500]
data_label_318 = data_label[3500:]
data_features = oversampling_data.drop(columns=10)
data_features_3500 = data_features[:3500]
data_features_318 = data_features[3500:]

# Get the numbers in array
tune_array_1_50 = list(range(1,51))
tune_array_2_5 = list(range(2,6))
tune_array_1_5 = list(range(1,6))

# build the RandomForestClassifier model
# The reason why I insert the hypeparameter "min_samples_leaf=1" and "min_samples_split=2" here
# is that I know they are the best hypeparameter after GridSearch method
decision_tree = RandomForestClassifier(min_samples_leaf=1,min_samples_split=2,random_state=0)

# use GridSearch to do hypeparameter tuning in order to get the best model
print("The program is doing hyperparameter tuning using Grid Search, please wait for two to three minutes...")
parameters = { 'max_depth':tune_array_1_50, 'min_samples_leaf':tune_array_1_5, 'min_samples_split':tune_array_2_5 }
clf = GridSearchCV(estimator = decision_tree, param_grid = parameters, scoring = 'f1', cv = 3, return_train_score = True, n_jobs=-1)
clf.fit(data_features_3500, data_label_3500)
print("best parameters are ", clf.best_params_)
label_predict = clf.predict(data_features_318)
print("length of label_predict:",len(label_predict))
print("f1 score (average = None) :", f1_score(data_label_318, label_predict, average=None))
print("f1 score (average = macro) :", f1_score(data_label_318, label_predict, average='macro'))
print("accuracy_score:", accuracy_score(data_label_318, label_predict))
print("confusion_matrix:\n", confusion_matrix(data_label_318, label_predict))

# Use the best model to predict the test file, get the result and output to txt file
label_predict = clf.predict(predict_features)
print("length of label_predict:",len(label_predict))
label_predict = pd.DataFrame(label_predict)
label_predict.to_csv('partA_label_predict.txt', index=False, header=None)
print("The program is finished. Please get the predict results in text file, 'partA_label_predict.txt', in the folder")