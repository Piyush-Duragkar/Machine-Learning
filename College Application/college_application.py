import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df=pd.read_csv("/content/College_Admissions.csv")
df.head()

new_df=df.drop(['Serial No.'],axis=1)
new_df.head()

print(new_df.shape)

# Checking for values which are empty or missing
new_df.isna().sum()

new_df.corr(method="pearson")

"""As We can See Chance of Admit is highly Correlated with GRE Score,Toefl Score and CGPA"""

# iloc is type of slicing, here we are including all rows and colums upto 6,excluding 7
X=new_df.iloc[:,:6]
y=new_df["Chance of Admit "]
new_df.head()

print(X.shape)
print(y.shape)
X.head()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.05,random_state=13)

from sklearn.linear_model import LinearRegression
#Linear Regression
Linear=LinearRegression()
Linear.fit(X_train,y_train)
y_pred=Linear.predict(X_test)
y_pred

from sklearn.metrics import mean_absolute_error,r2_score
print("the accuracy or the R2 score of the model is ",r2_score(y_pred,y_test))
print("mean_absolute_error  of the model is ",mean_absolute_error(y_pred,y_test))

"""We have used the mean absolute eror because we are dealing with a continuous value which is suitable for the mean absolute error.
methods like precision or f1 would have been opted if the output was related to the classification amongst categories.

"""

