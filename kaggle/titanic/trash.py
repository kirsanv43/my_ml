import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

gender_df= pd.read_csv('../input/titanic/gender_submission.csv')
# test_df = pd.read_csv('../input/titanic/test.csv')
train_df = pd.read_csv('../input/titanic/train.csv')
print(train_df.isnull().sum(axis = 0))


from sklearn.impute import SimpleImputer
si = SimpleImputer(strategy="most_frequent")
inputerResult = si.fit_transform(train_df[['Embarked']])

train_df['Embarked'] = inputerResult

si = SimpleImputer(strategy="mean")
inputerResult = si.fit_transform(train_df[['Age']])
train_df['Age'] = inputerResult
train_df["Age"] = train_df['Age'].map(lambda x: '{:.2f}'.format(float(x)))

qwe = train_df[train_df['Sex']=='female']


qwe = qwe.groupby(["Age"]).agg({'Survived': 'count'}).reset_index()
print(qwe.head())
fig, ax = plt.subplots(figsize=(50,10))

ax = sns.barplot(x="Age", y="Survived", data=qwe, ax=ax)