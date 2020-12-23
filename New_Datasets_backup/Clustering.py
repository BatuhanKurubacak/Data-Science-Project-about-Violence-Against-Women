# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt



data=pd.read_csv("age_data_value.csv")

data_c=data.copy()

#print(data_c['Age'].unique()) #Now I know how many unique values?

data_c.Age=data_c['Age'].str.replace('15-24','1')
data_c.Age=data_c['Age'].str.replace('25-34','2')
data_c.Age=data_c['Age'].str.replace('35-49','3')

data_c.Gender=data_c['Gender'].str.replace('F','1')
data_c.Gender=data_c['Gender'].str.replace('M','0')




data_c.Gender=data_c.Gender.apply(int)
data_c.Age= data_c.Age.apply(int)

data_c=data_c.drop(['Country'],1)
data_c=data_c.drop(['RecordID'],1)
data_c=data_c.drop(['Value'],1)
data_c['Question Type']=data_c['Question Type'].apply(int)


X = np.array(data_c.drop(['Age'], 1).astype(float))
#X = np.array(data_c['Value']).reshape(-1,1)
y = np.array(data_c['Age'])
#print(data_c.info())

kmeans = KMeans(n_clusters=3) # You want cluster the passenger records into 2: Survived or Not survived
#kmeans.fit(X)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)


predict=kmeans.predict(X)
v=X_scaled


correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1


print(correct/len(X))

print(kmeans.cluster_centers_)

'''

plt.scatter(X[ : , 0], X[ : , 1], s =10, c='b')
plt.scatter(1, 0.5, s=200, c='g', marker='s')
plt.scatter(0, 0.2, s=200, c='r', marker='s')
plt.scatter(0, 0.8, s=200, c='y', marker='s')
plt.show()
'''


'''
plt.scatter(v[predict==0,0],v[predict==0,2],s=300,color='red')
plt.scatter(v[predict==2,0],v[predict==2,2],s=300,color='blue')
plt.scatter(v[predict==4,0],v[predict==4,2],s=300,color='green')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label = 'Centroids')
plt.title('Clusters of Age')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100')
plt.show()
'''