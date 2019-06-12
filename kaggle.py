from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np

def encoding(z):
    columns = z.columns.values
    for column in columns:
        if z[column].dtype != np.int64 and z[column].dtype != np.float64:
            label = preprocessing.LabelEncoder()
            z[column] = label.fit_transform(z[column])
    return z

def pre(z):
    df = z.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df)
    z = pd.DataFrame(x_scaled)
    return z

# Reading Data
data = pd.read_csv("train.csv")
testdata = pd.read_csv("test.csv")
x = data.drop(["PassengerId","Survived","Name","Ticket","Cabin","Embarked"],1)
x_test = testdata.drop(["PassengerId","Name","Ticket","Cabin","Embarked"],1)

# Filling Null Values
x.fillna(value=x.mean(),inplace=True)
x_test.fillna(value=x.mean(),inplace=True)
# x["Embarked"].fillna(value="S",inplace=True)
# x_test["Embarked"].fillna(value="S",inplace=True)

# Encoding Non-Numerical Data
x = encoding(x)
x_test = encoding(x_test)

# Preprocessing the Data
x = pre(x)
x_test = pre(x_test)

# Making the model using Gradient Boosting Algorithm
y = data["Survived"]
model = GradientBoostingClassifier(n_estimators = 50)
model.fit(x,y)
y_pred = model.predict(x_test)
answer = pd.DataFrame(testdata.iloc[:,0])
answer["Survived"] = y_pred

# Exporting the answer
export_csv = answer.to_csv (r"export_dataframe.csv", index = None, header=True)