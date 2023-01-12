

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

"""## Getting the dataset"""

df=pd.read_csv('NASA.csv')

df.head()

df.shape

"""## Understanding three different features"""

df['absolute_magnitude'].describe()

df['relative_velocity'].describe()

df['miss_distance'].describe()

"""## Data Processing """

# transforming True and False to 1's and 0's for testing our model

df['hazardous']=df['hazardous'].replace({True:1,False:0})
df['sentry_object']=df['sentry_object'].replace({True:1,False:0})

df.isna().sum()

df.head()

"""## Box Plot"""

box_feature = ["est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance", "sentry_object", "absolute_magnitude"]
for i in box_feature:
    df.boxplot(by ='hazardous', column =i, grid = False)

"""## Min-Max Scaling"""

df_new = df.drop(['id','name','orbiting_body'], axis=1)

scaler = MinMaxScaler()
 
df_scaled = scaler.fit_transform(df_new.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=["est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance", "sentry_object", "absolute_magnitude", "hazardous"])
 
print("Scaled Dataset Using MinMaxScaler")
df_scaled.head()

"""## Outlier Removal"""

def outlier_treatment(datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn , [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range,upper_range

    
for feature in ["est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance", "absolute_magnitude"]:
    lowerbound, upperbound = outlier_treatment(df_scaled[feature])
    print("For Column: ", feature, " ->  Range:", lowerbound, " To ", upperbound)
    print("Total outliers removed: ", len(df_scaled[(df_scaled[feature] < lowerbound) | (df_scaled[feature] > upperbound)]))
    print()
    values = df_scaled[(df_scaled[feature] < lowerbound) | (df_scaled[feature] > upperbound)]
    df_scaled.drop(values.index, inplace=True)

"""## Pair Plots"""

sns.pairplot(df_scaled, hue="hazardous", vars=["est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance", "sentry_object", "absolute_magnitude"])

"""## SVM"""

train_data, test_data, y_train, y_test = train_test_split(df_scaled.drop(columns="hazardous"), df_scaled.hazardous, test_size=0.2, random_state=4)

from sklearn import svm
from sklearn import metrics

model = svm.SVC(kernel='rbf', C=10, gamma=1, class_weight="balanced")
model.fit(train_data, y_train)
y_pred = model.predict(test_data)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))

print ("F1_Score:",metrics.f1_score(y_test, y_pred))

"""## PCA"""

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df_scaled)


pca = PCA(n_components = 2)
pca.fit(train_data)
data_pca_train = pca.transform(train_data)
data_pca_test = pca.transform(test_data)
data_pca = pd.DataFrame(data_pca_train,columns=['PC1','PC2'])

from sklearn import svm
from sklearn import metrics

model = svm.SVC(kernel='rbf', C=10, gamma=1)
model.fit(data_pca_train, y_train)
y_pred_1 = model.predict(data_pca_test)

print ("Accuracy:",metrics.accuracy_score(y_test, y_pred_1))

"""## Grid Search"""

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [1, 2, 10], 
              'gamma': [1, 0.01],
              'kernel': ['rbf', 'poly']} 
  
grid = GridSearchCV(svm.SVC(class_weight="balanced"), param_grid, refit = True, verbose = 3, n_jobs=-1, cv=2)

grid.fit(train_data, y_train)

grid.best_estimator_

"""## Random Forest """

model = RandomForestClassifier()
model.fit(train_data,y_train)
y_pred = model.predict(test_data)

accuracy = model.score(train_data, y_train)

print(accuracy)

"""## KNN"""

from sklearn.neighbors import KNeighborsClassifier

  
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
  
# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, y_train)
      
    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(train_data, y_train)
  
# Generate plot
plt.plot(neighbors, train_accuracy, label = 'Testing dataset Accuracy')
  
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()







