import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH")

train_data = pd.read_csv(TRAIN_DATA_PATH)
X_train, y_train = train_data.iloc[:,:-1], train_data.iloc[:,-1]

level_0 = list()
level_0.append(('RF', RandomForestClassifier(n_estimators=700)))
level_0.append(('LR',LogisticRegression(max_iter=6000)))
        
level_1 = SVC(C=1.2)
model = StackingClassifier(estimators=level_0, final_estimator=level_1, cv=4)
model.fit(X_train, y_train)

test_data = pd.read_csv(TEST_DATA_PATH)
submission = model.predict(test_data)
submission = pd.DataFrame(submission)

submission.to_csv('submission.csv', header=['class'], index=False)

