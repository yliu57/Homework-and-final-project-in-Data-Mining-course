import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Read the train and test dataset
data = pd.read_csv('./train_B.csv', header = 0)
predict_features = pd.read_csv('./test_B.csv', header = 0)

# ------------------ Data preprocessing for test dataset

# change the features' name to number
predict_features = predict_features.to_numpy()
predict_features = pd.DataFrame(predict_features)

# select the column of c_charge_desc
predict_features_text = predict_features[[9]]
predict_features_text = predict_features_text.to_numpy()
predict_features_text = pd.DataFrame(predict_features_text)

# select the column of sex, age_cat, race, c_charge_degree
predict_features_ohe = predict_features[[0,2,3,8]]
predict_features_ohe = predict_features_ohe.to_numpy()
predict_features_ohe = pd.DataFrame(predict_features_ohe)

# select the column of age, juv_fel_count, juv_misd_count, juv_other_count, priors_count
predict_features_num = predict_features[[1,4,5,6,7]]
predict_features_num = predict_features_num.to_numpy()
predict_features_num = pd.DataFrame(predict_features_num)

# ------------------ Data preprocessing for train dataset

# Get the rows whose features in 'label' is 0
data_0 = data[data['label']==0]
# Get the rows whose features in 'label' is 1
data_1 = data[data['label']==1]

# Find that the quantity of labels in 1 (2286) is smaller than the quantity of labels in 0 (2763)
# Oversample the data_1, the number of 1 changes from 2286 to 2763
oversampling_data_1 = np.append(data_1, data_1, axis=0)
oversampling_data_1 = oversampling_data_1[:2763]

# Add data_0 and oversampling_data_1 together and refresh the row number
oversampling_data = np.append(data_0, oversampling_data_1, axis=0)
oversampling_data = pd.DataFrame(oversampling_data)

# shuffle the oversampling_data and refresh the row number
oversampling_data = shuffle(oversampling_data)
oversampling_data = oversampling_data.to_numpy()
oversampling_data = pd.DataFrame(oversampling_data)

# Get the label of oversampling_data which is used for training and validation
data_label = oversampling_data[10]
data_label = data_label.astype(int)

# Get the features columns of oversampling_data which is used for training and validation
data_features = oversampling_data.drop(columns=10)

# select the column of c_charge_desc
data_features_text = data_features[[9]]
data_features_text = data_features_text.to_numpy()
data_features_text = pd.DataFrame(data_features_text)

# select the column of sex, age_cat, race, c_charge_degree
data_features_ohe = data_features[[0,2,3,8]]
data_features_ohe = data_features_ohe.to_numpy()
data_features_ohe = pd.DataFrame(data_features_ohe)

# select the column of age, juv_fel_count, juv_misd_count, juv_other_count, priors_count
data_features_num = data_features[[1,4,5,6,7]]
data_features_num = data_features_num.to_numpy()
data_features_num = pd.DataFrame(data_features_num)

# ------------------ OneHotEncoder for train and test dataset
# Use OneHotEncoder to change the categorical features to numeric features
# focusing on the categorical features(column:[0,2,3,8]) in train and test dataset
# If one object has this numeric feature, its numeric feature value will be 1, otherwise it will be 0
enc = OneHotEncoder(dtype=int)
enc.fit_transform(data_features_ohe)
print("enc.categories_:",enc.categories_)

# OneHotEncoder for the train dataset
enc_data_features = enc.transform(data_features_ohe).toarray()
enc_data_features = pd.DataFrame(enc_data_features)
print('enc_data_features:', enc_data_features)

# OneHotEncoder for the test dataset
enc_predict_features = enc.transform(predict_features_ohe).toarray()
enc_predict_features = pd.DataFrame(enc_predict_features)
print('enc_predict_features:', enc_predict_features)

# Use OneHotEncoder to change the text features to numeric features
# focusing on the text features(column:[9]) in train and test dataset
# If one object has this numeric feature, its numeric feature value will be 1, otherwise it will be 0
total_features_text = np.append(data_features_text, predict_features_text, axis=0)
enc = OneHotEncoder(dtype=int)# dtype=int
enc.fit(total_features_text)
print("enc.categories_:",enc.categories_)

# OneHotEncoder for the train dataset
enc_data_features_text = enc.transform(data_features_text).toarray()
enc_data_features_text = pd.DataFrame(enc_data_features_text)
print('data_features_text:', enc_data_features_text)

# OneHotEncoder for the test dataset
enc_predict_features_text = enc.transform(predict_features_text).toarray()
enc_predict_features_text = pd.DataFrame(enc_predict_features_text)
print('predict_features_text:', enc_predict_features_text)

# ------------------ Data preprocessing for Decision tree and Grid Search
# Combine the columns in different features together
# number features: column[1,4,5,6,7]
# categorical features: column:[0,2,3,8]
# text features: column:[9]

# Combine the columns for the train dataset
enc_data_features = np.append(enc_data_features, enc_data_features_text, axis=1)
enc_data_features = np.append(enc_data_features, data_features_num, axis=1)
enc_data_features = pd.DataFrame(enc_data_features)

# Combine the columns for the test dataset
enc_predict_features = np.append(enc_predict_features, enc_predict_features_text, axis=1)
enc_predict_features = np.append(enc_predict_features, predict_features_num, axis=1)
enc_predict_features = pd.DataFrame(enc_predict_features)

# Split the oversampling train dataset to two parts
# One part has 5000 data for training and the other part has 526 data for validation

# split the features
enc_data_features_5000 = enc_data_features[:5000]
enc_data_features_526 = enc_data_features[5000:]

# split the labels
data_label_5000 = data_label[:5000]
data_label_526 = data_label[5000:]

# ------------------ Decision Tree implements and Grid Search approaches for hyperparameter tuning
# Build the decision tree model
decision_tree = DecisionTreeClassifier()

# Grid Search approaches for hyperparameter tuning
print("The program is doing hyperparameter tuning using Grid Search, please wait for three to five minutes...")
tune_array_1_20 = list(range(1,21))
tune_array_2_5 = list(range(2,6))
tune_array_1_10 = list(range(1,11))
parameters = { 'max_depth':tune_array_1_20, 'min_samples_leaf':tune_array_1_10,'min_samples_split':tune_array_2_5 }
clf = GridSearchCV(estimator = decision_tree, param_grid = parameters, scoring = 'accuracy', cv = 3, return_train_score = True, n_jobs=-1)
clf.fit(enc_data_features_5000, data_label_5000)
print("best parameters are ", clf.best_params_)
label_predict = clf.predict(enc_data_features_526)
print("length of label_predict:",len(label_predict))
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
print("accuracy_score:", accuracy_score(data_label_526, label_predict))
print("confusion_matrix:", confusion_matrix(data_label_526, label_predict))

# Use the best training and validation model to predict the features in the test dataset
# Output the predict results to the text file 'label_predict.txt'
label_predict = clf.predict(enc_predict_features)
print("length of label_predict:",len(label_predict))
label_predict = pd.DataFrame(label_predict)
label_predict.to_csv('partB_label_predict.txt', index=False, header=None)
print("The program is finished. Please get the predict results in text file, 'partB_label_predict.txt', in the folder")