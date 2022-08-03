import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

heart_disease = pd.read_csv('heart.csv')
print(heart_disease.head())
print(heart_disease.info())  # no null values in dataset

lab_enc = LabelEncoder()  # Labelling all non-numeric data fields into nominal numbers
heart_disease['Sex'] = lab_enc.fit_transform(heart_disease['Sex'])
heart_disease['ChestPainType'] = lab_enc.fit_transform(heart_disease['ChestPainType'])
heart_disease['RestingECG'] = lab_enc.fit_transform(heart_disease['RestingECG'])
heart_disease['ExerciseAngina'] = lab_enc.fit_transform(heart_disease['ExerciseAngina'])
heart_disease['ST_Slope'] = lab_enc.fit_transform(heart_disease['ST_Slope'])

print(heart_disease.head())

X = heart_disease.drop(columns='HeartDisease')
Y = heart_disease['HeartDisease']

model = LogisticRegression(max_iter=1000)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y,random_state=42)
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
train_acc = accuracy_score(X_train_prediction, Y_train)
print("Training accuracy: ", train_acc)
print('-----------------------------------------------------------------------')

X_test_prediction = model.predict(X_test)
test_acc = accuracy_score(X_test_prediction, Y_test)
print("Testing accuracy: ", test_acc)
print('-----------------------------------------------------------------------')

print(Y_test.value_counts())
cm = confusion_matrix(X_test_prediction, Y_test)
print(cm)
print('-----------------------------------------------------------------------')


print("Model of Random Forest Classifier")
model = RandomForestClassifier()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
train_acc = accuracy_score(X_train_prediction, Y_train)
print("Training accuracy: ", train_acc)
X_test_prediction = model.predict(X_test)
test_acc = accuracy_score(X_test_prediction, Y_test)
print("Testing accuracy: ", test_acc)

eg1 = (40, 1, 1, 130, 305, 0, 1, 142, 1, 1.2, 1)
eg1 = np.asarray(eg1)
eg1 = eg1.reshape(1, -1)
eg1_prediction = model.predict(eg1)
print("Heart Issue!!!") if eg1_prediction==1 else print("No worries:)")
