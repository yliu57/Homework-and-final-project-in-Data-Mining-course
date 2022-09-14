import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# ---------------- Data preprocessing
data = pd.read_csv('./train.csv', header = 0)

race_AA = data[data['race']=='African-American'].index
race_CA = data[data['race']=='Caucasian'].index

data_label = data[['label']]

data_features = data.drop(columns='label')

data_features_text = data_features[['c_charge_desc']]

data_features_ohe_withrace = data_features[['sex','age_cat','c_charge_degree','race']]
data_features_ohe_withoutrace = data_features[['sex','age_cat','c_charge_degree']]

data_features_num = data_features[['age','juv_fel_count','juv_misd_count','juv_other_count','priors_count']]

# ----------------- Use OneHotEncoder to deal with categorical data
# from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(dtype=int)

enc.fit_transform(data_features_ohe_withrace)
# print("enc.categories_:",enc.categories_)
enc_data_features_withrace = enc.transform(data_features_ohe_withrace).toarray()
enc_data_features_withrace = pd.DataFrame(enc_data_features_withrace)
# print('enc_data_features:', enc_data_features_withrace)

enc.fit_transform(data_features_ohe_withoutrace)
# print("enc.categories_:",enc.categories_)
enc_data_features_withoutrace = enc.transform(data_features_ohe_withoutrace).toarray()
enc_data_features_withoutrace = pd.DataFrame(enc_data_features_withoutrace)
# print('enc_data_features:', enc_data_features_withoutrace)

# ----------------- Use OneHotEncoder to deal with text
# from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(dtype=int)
enc.fit(data_features_text)
# print("enc.categories_:",enc.categories_)
enc_data_features_text = enc.transform(data_features_text).toarray()
enc_data_features_text = pd.DataFrame(enc_data_features_text)
# print('data_features_text:', enc_data_features_text)

# ----------------- Combine the num, the categorical features and text together
enc_data_features_withrace = np.append(enc_data_features_withrace, enc_data_features_text, axis=1)
enc_data_features_withrace = np.append(enc_data_features_withrace, data_features_num, axis=1)
enc_data_features_withrace = pd.DataFrame(enc_data_features_withrace)


enc_data_features_withoutrace = np.append(enc_data_features_withoutrace, enc_data_features_text, axis=1)
enc_data_features_withoutrace = np.append(enc_data_features_withoutrace, data_features_num, axis=1)
enc_data_features_withoutrace = pd.DataFrame(enc_data_features_withoutrace)

# -------------------- Import the decision tree model and use the approach of cross validation
decision_tree = DecisionTreeClassifier()

# Get the model after cross validation
prediction_withrace = cross_val_predict(decision_tree, enc_data_features_withrace, data_label, cv=5)
prediction_withoutrace = cross_val_predict(decision_tree, enc_data_features_withoutrace, data_label, cv=5)

# -------------------- Calculate the false positive rate and true positive rate of all races
print("\nAll races  -----------------------------------------------------------------")
print("\nQuestion 1 -----------------------------------------------------------------")
# In confusion matrix, the count of true negatives is C(0,0); the count of false positives is(0,1)
# Related sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

print("f1 score average = None (with race):", f1_score(data_label, prediction_withrace, average=None))
print("f1 score average = macro (with race):", f1_score(data_label, prediction_withrace, average='macro'))
print("accuracy_score (with race):", accuracy_score(data_label, prediction_withrace))
print("confusion_matrix (with race):", confusion_matrix(data_label, prediction_withrace))
cm_withrace = confusion_matrix(data_label, prediction_withrace)
print("fp rate (with race):",cm_withrace[0][1]/(cm_withrace[0][1]+cm_withrace[0][0]))

print("-------------------------------")

print("f1 score average = None (without race):", f1_score(data_label, prediction_withoutrace, average=None))
print("f1 score average = macro (without race):", f1_score(data_label, prediction_withoutrace, average='macro'))
print("accuracy_score (without race):", accuracy_score(data_label, prediction_withoutrace))
print("confusion_matrix (without race):", confusion_matrix(data_label, prediction_withoutrace))
cm_withoutrace = confusion_matrix(data_label, prediction_withoutrace)
print("fp rate (without race):",cm_withoutrace[0][1]/(cm_withoutrace[0][1]+cm_withoutrace[0][0]) )

print("\nQuestion 2 -----------------------------------------------------------------")
# Predicted Positive and True Positive / Predicted Positive
# Predicted Positive and True Positive = True Positive, C(1,1)
# Predicted Positive = True Positive + False Postive, C(1,1) + C(0,1)

print("f1 score average = None (with race):", f1_score(data_label, prediction_withrace, average=None))
print("f1 score average = macro (with race):", f1_score(data_label, prediction_withrace, average='macro'))
print("accuracy_score (with race):", accuracy_score(data_label, prediction_withrace))
print("confusion_matrix (with race):", confusion_matrix(data_label, prediction_withrace))
cm_withrace = confusion_matrix(data_label, prediction_withrace)
print("[Predicted Positive and True Positive / Predicted Positive] rate (with race):",cm_withrace[1][1]/(cm_withrace[1][1]+cm_withrace[0][1]))

print("-------------------------------")

print("f1 score average = None (without race):", f1_score(data_label, prediction_withoutrace, average=None))
print("f1 score average = macro (without race):", f1_score(data_label, prediction_withoutrace, average='macro'))
print("accuracy_score (without race):", accuracy_score(data_label, prediction_withoutrace))
print("confusion_matrix (without race):", confusion_matrix(data_label, prediction_withoutrace))
cm_withoutrace = confusion_matrix(data_label, prediction_withoutrace)
print("[Predicted Positive and True Positive / Predicted Positive] rate (without race):",cm_withoutrace[1][1]/(cm_withoutrace[1][1]+cm_withoutrace[0][1]))


# -------------------- Calculate the false positive rate and true positive rate of African-American
print("\n\nAfrican-American  ----------------------------------------------------------")
print("\nQuestion 1 -----------------------------------------------------------------")
print("False positive, African-American, with race:\n")
fp_AA = 0
AN_AA = 0 # Actual negative
for index in race_AA:
    if(data_label.iat[index,0] == 0 and prediction_withrace[index] == 1):
        fp_AA+=1
    if(data_label.iat[index,0] == 0):
        AN_AA+=1
print("fp_AA:",fp_AA)
print("Actual negative:",AN_AA)
print("fp rate:",fp_AA/AN_AA)

print("-------------------------------")

print("False positive, African-American , without race:\n")
fp_AA = 0
AN_AA = 0 # Actual negative
for index in race_AA:
    if(data_label.iat[index,0] == 0 and prediction_withoutrace[index] == 1):
        fp_AA+=1
    if(data_label.iat[index,0] == 0):
        AN_AA+=1
print("fp_AA:",fp_AA)
print("Actual negative:",AN_AA)
print("fp rate:",fp_AA/AN_AA)


print("\nQuestion 2 -----------------------------------------------------------------")
print("False positive, African-American, with race:\n")
pptp_AA = 0 # Predicted Positive and True Positive
pp_AA = 0 # Predicted Positive
for index in race_AA:
    if(data_label.iat[index,0] == 1 and prediction_withrace[index] == 1):
        pptp_AA+=1
    if(prediction_withrace[index] == 1):
        pp_AA+=1
print("Predicted Positive and True Positive (AA):",pptp_AA)
print("Predicted Positive (AA):",pp_AA)
print("[Predicted Positive and True Positive / Predicted Positive] rate:",pptp_AA/pp_AA)

print("-------------------------------")

print("False positive, African-American , without race:\n")
pptp_AA = 0
pp_AA = 0 # Actual negative
for index in race_AA:
    if(data_label.iat[index,0] == 1 and prediction_withoutrace[index] == 1):
        pptp_AA+=1
    if(prediction_withoutrace[index] == 1):
        pp_AA+=1
print("Predicted Positive and True Positive (AA):",pptp_AA)
print("Predicted Positive (AA):",pp_AA)
print("[Predicted Positive and True Positive / Predicted Positive] rate:",pptp_AA/pp_AA)


# -------------------- Calculate the false positive rate and true positive rate of Caucasian
print("\n\nCaucasian  -----------------------------------------------------------------")
print("\nQuestion 1 -----------------------------------------------------------------")
print("False positive, Caucasian, with race:\n")
fp_CA = 0
AN_CA = 0 # Actual negative
for index in race_CA:
    if(data_label.iat[index,0] == 0 and prediction_withrace[index] == 1):
        fp_CA+=1
    if(data_label.iat[index,0] == 0):
        AN_CA+=1
print("fp_CA:",fp_CA)
print("Actual negative:",AN_CA)
print("fp rate:",fp_CA/AN_CA)

print("-------------------------------")

print("False positive, Caucasian, with out race:\n")
fp_CA = 0
AN_CA = 0 # Actual negative
for index in race_CA:
    if(data_label.iat[index,0] == 0 and prediction_withoutrace[index] == 1):
        fp_CA+=1
    if(data_label.iat[index,0] == 0):
        AN_CA+=1
print("fp_CA:",fp_CA)
print("Actual negative:",AN_CA)
print("fp rate:",fp_CA/AN_CA)



print("\nQuestion 2 -----------------------------------------------------------------")
print("False positive, Caucasian, with race:\n")
pptp_CA = 0 # Predicted Positive and True Positive
pp_CA = 0 # Predicted Positive
for index in race_CA:
    if(data_label.iat[index,0] == 1 and prediction_withrace[index] == 1):
        pptp_CA+=1
    if(prediction_withrace[index] == 1):
        pp_CA+=1
print("Predicted Positive and True Positive (AA):",pptp_CA)
print("Predicted Positive (AA):",pp_CA)
print("[Predicted Positive and True Positive / Predicted Positive] rate:",pptp_CA/pp_CA)

print("-------------------------------")

print("False positive, Caucasian, with out race:\n")
ppta_CA = 0
pp_CA = 0 # Actual negative
for index in race_CA:
    if(data_label.iat[index,0] == 1 and prediction_withoutrace[index] == 1):
        ppta_CA+=1
    if(prediction_withoutrace[index] == 1):
        pp_CA+=1
print("Predicted Positive and True Positive (AA):",pptp_CA)
print("Predicted Positive (AA):",pp_CA)
print("[Predicted Positive and True Positive / Predicted Positive] rate:",pptp_CA/pp_CA)



