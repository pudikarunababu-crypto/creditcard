 import pandas as pd
 import numpy as np

data=pd.read_csv(path + "/creditcard.csv")

print(data.head())

print(data.shape)

print(data.info())

print(data.describe())

print(data['Class'].value_counts())

import matplotlib.pyplot as plt
data['Class'].value_counts().plot(kind='bar')
plt.title("Before SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

#pip install --upgrade scikit-learn imbalanced-learn==>in case error run this Line
from imblearn . over_sampling import SMOTE
X = data.drop('Class', axis=1)
Y = data['Class']
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, Y)

pd.Series(y_res).value_counts().plot(kind='bar')
plt.title("After SMOTE")
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report,accuracy_score
print("Accuracy:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

import numpy as np
normal=X[X.index.isin(data[data['Class']==0].index)].iloc[0]
print("Prediction:",model.predict([normal]))

fraud=X[X.index.isin(data[data['Class']==1].index)].iloc[0]
print("Prediction:",model.predict([fraud]))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
tree=model.estimators_[0]
plt.figure(figsize=(20,10))
plot_tree(tree,feature_names=X.columns,class_names=['Normal','Fraud'],filled=True)
plt.show()
