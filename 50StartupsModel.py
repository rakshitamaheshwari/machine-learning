
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder    #data preprocessing
from sklearn.preprocessing import OneHotEncoder   #data preprocessing
from sklearn import metrics


dataset=pd.read_csv('Desktop/50_Startups.csv')
y=dataset.iloc[:,-1]
X=dataset.iloc[:,:4]
encode_x=LabelEncoder()                          #provide encoding of string col that is state
x4=X.iloc[:,-1]
x4_new=encode_x.fit_transform(x4)                #x me states ko fit krke transform
X.iloc[:,-1]=x4_new

#one hot encoding
#creates new fields/columns to remove concepts of priority according to no.encoding

oneHotEncoding = OneHotEncoder(categorical_features=[-1])
new=oneHotEncoding.fit_transform(X).toarray()

X=new[:,1:]                                      #removed one dummy var to remove dummy var trap.
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


model=LinearRegression()                         #model training
model.fit(X_train,y_train)
y_pred=model.predict(X_test)                       #testing

model.coef_
model.intercept_

metrics.mean_absolute_error(y_test,y_pred)

