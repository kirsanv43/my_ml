import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Data.csv') 
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean')
missingvalues = missingvalues.fit(X[:, 1:3])
X[:, 1:3] = missingvalues.transform(X[:, 1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([("Name_Of_Your_Step", OneHotEncoder(),[0])], remainder="passthrough") # The last arg ([0]) is the list of columns you want to transform in this step
#ct.fit_transform(X) 

#labelEncoder_X = LabelEncoder()
#X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])

X = np.array(ct.fit_transform(X), dtype=np.float)

Y = LabelEncoder().fit_transform(Y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = 0.8, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))