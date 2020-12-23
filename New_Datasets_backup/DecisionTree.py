# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from six import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


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
data_c['Question Type']=data_c['Question Type'].apply(int)



feature_cols = [ 'Value','Age']
X = data_c[feature_cols] # Features
y = data_c.Gender # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="gini",splitter='best',max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Accuracy, how often is the classifier correct?

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['Male','Female'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('AgeDecisionTree.png')
#Image(graph.create_png())



data=pd.read_csv('edu_data.csv')

data_c1=data.copy()


print(data_c1['Education'].unique())

data_c1.Education=data_c1['Education'].str.replace('Primary','1')
data_c1.Education=data_c1['Education'].str.replace('Secondary','2')
data_c1.Education=data_c1['Education'].str.replace('Higher','3')
data_c1.Education=data_c1['Education'].str.replace('No education','0')



data_c1.Gender=data_c1['Gender'].str.replace('F','1')
data_c1.Gender=data_c1['Gender'].str.replace('M','0')

data_c1.Gender=data_c1.Gender.apply(int)
data_c1.Education= data_c1.Education.apply(int)
data_c1['Question Type']=data_c1['Question Type'].apply(int)

feature1_cols = [ 'Value','Education']
X = data_c1[feature1_cols] # Features
y = data_c1.Gender # Target variable

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy",splitter='best',max_depth=4)

# Train Decision Tree Classifer
clf = clf.fit(X_train1,y_train1)

#Predict the response for test dataset
y_pred1 = clf.predict(X_test1)

print("Accuracy:",metrics.accuracy_score(y_test1, y_pred1))
# Model Accuracy, how often is the classifier correct?

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature1_cols,class_names=['Male','Female'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('EduDecisionTree.png')
#Image(graph.create_png())






data=pd.read_csv('emp_data.csv')

data_c2=data.copy()


print(data_c2['Employment'].unique())


data_c2.Employment=data_c2['Employment'].str.replace('Employed for kind','1')
data_c2.Employment=data_c2['Employment'].str.replace('Unemployed','2')
data_c2.Employment=data_c2['Employment'].str.replace('Employed for cash','3')



data_c2.Gender=data_c2['Gender'].str.replace('F','1')
data_c2.Gender=data_c2['Gender'].str.replace('M','0')

data_c2.Gender=data_c2.Gender.apply(int)
data_c2.Employment= data_c2.Employment.apply(int)
data_c2['Question Type']=data_c2['Question Type'].apply(int)


feature2_cols = [ 'Value','Employment']
X = data_c2[feature2_cols] # Features
y = data_c2.Gender # Target variable

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


clf = DecisionTreeClassifier(criterion="entropy",splitter='random',max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train2,y_train2)

#Predict the response for test dataset
y_pred2 = clf.predict(X_test2)

print("Accuracy:",metrics.accuracy_score(y_test2, y_pred2))
# Model Accuracy, how often is the classifier correct?

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature2_cols,class_names=['Male','Female'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('EmploymentDecisionTree.png')
#Image(graph.create_png())





data=pd.read_csv('marital_data.csv')

data_c3=data.copy()


print(data_c3['Marital Status'].unique())


data_c3['Marital Status']=data_c3['Marital Status'].str.replace('Never married','1')
data_c3['Marital Status']=data_c3['Marital Status'].str.replace('Widowed, divorced, separated','2')
data_c3['Marital Status']=data_c3['Marital Status'].str.replace('Married or living together','3')


data_c3.Gender=data_c3['Gender'].str.replace('F','1')
data_c3.Gender=data_c3['Gender'].str.replace('M','0')

data_c3.Gender=data_c3.Gender.apply(int)
data_c3['Marital Status']= data_c3['Marital Status'].apply(int)
data_c3['Question Type']=data_c3['Question Type'].apply(int)

feature3_cols = [ 'Value','Marital Status']
X = data_c3[feature3_cols] # Features
y = data_c3.Gender # Target variable


X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


clf = DecisionTreeClassifier(criterion="gini",splitter='random',max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train3,y_train3)

#Predict the response for test dataset
y_pred3 = clf.predict(X_test3)

print("Accuracy:",metrics.accuracy_score(y_test3, y_pred3))
# Model Accuracy, how often is the classifier correct?

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature3_cols,class_names=['Male','Female'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('MaritalDecisionTree.png')
#Image(graph.create_png())



data=pd.read_csv('res_data.csv')

data_c4=data.copy()


print(data_c4['Residence'].unique())

data_c4['Residence']=data_c4['Residence'].str.replace('Rural','1')
data_c4['Residence']=data_c4['Residence'].str.replace('Urban','0')



data_c4.Gender=data_c4['Gender'].str.replace('F','1')
data_c4.Gender=data_c4['Gender'].str.replace('M','0')

data_c4.Gender=data_c4.Gender.apply(int)
data_c4['Residence']= data_c4['Residence'].apply(int)
data_c4['Question Type']=data_c4['Question Type'].apply(int)



feature4_cols = [ 'Value','Gender']
X = data_c4[feature4_cols] # Features
y = data_c4.Residence # Target variable


X_train4, X_test4, y_train4, y_test4 = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test



clf = DecisionTreeClassifier(criterion="gini",splitter='random',max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train4,y_train4)

#Predict the response for test dataset
y_pred4 = clf.predict(X_test4)

print("Accuracy:",metrics.accuracy_score(y_test4, y_pred4))
# Model Accuracy, how often is the classifier correct?

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature3_cols,class_names=['Urban','Rural'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('ResidenceDecisionTree.png')
#Image(graph.create_png())




'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Create a random dataset

X = data_c.iloc[:,3].values.reshape(-1,1)
y = data_c.iloc[:,5].values.reshape(-1,1)


# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_3 = DecisionTreeRegressor(max_depth=8)
regr_1.fit(X, y)
regr_2.fit(X, y)
regr_3.fit(X, y)

# Predict

y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
y_3 = regr_3.predict(X_test)

# Plot the results
plt.figure()
s = 25
#plt.scatter(y[:, 0], y[:, 1], c="navy", s=s,
#            edgecolor="black", label="data")
plt.scatter(X_test, y_1, c="cornflowerblue", s=s,
            edgecolor="black", label="max_depth=2")
plt.scatter(X_test, y_2, c="red", s=s,
            edgecolor="black", label="max_depth=5")
plt.scatter(X_test, y_3, c="orange", s=s,
            edgecolor="black", label="max_depth=8")
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.xlabel("target 1")
plt.ylabel("target 2")
plt.title("Multi-output Decision Tree Regression")
plt.legend(loc="best")
plt.show()
'''

'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# Create the dataset
rng = np.random.RandomState(1)
X = data_c.iloc[:,3].values.reshape(-1,1)
y = data_c.iloc[:,5].values.reshape(-1,1)

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=4)

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)

regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
y_1 = regr_1.predict(X)
y_2 = regr_2.predict(X)

# Plot the results
plt.figure()
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()

'''
'''
from sklearn import linear_model
from sklearn import svm


classifiers = [DecisionTreeRegressor(),
               svm.SVR(),
               linear_model.SGDRegressor(),
               linear_model.BayesianRidge(),
               linear_model.LassoLars(),
               linear_model.ARDRegression(),
               linear_model.PassiveAggressiveRegressor(),
               linear_model.TheilSenRegressor(),
               linear_model.LinearRegression()]    




for item in classifiers:
    print(item)
    clf = item
    clf.fit(X_train,y_train)
    print(clf.predict(X_test),'\n')
'''

'''
age=data_c.iloc[:,3].values.reshape(-1,1)
question=data_c.iloc[:,4].values.reshape(-1,1)
value=data_c.iloc[:,5].values.reshape(-1,1)

regression_tree=DecisionTreeRegressor()
regression_tree.fit(age,value)

#print(regression_tree.predict([[1]]))

plt.scatter(age,value,color='red')
x=np.arange(min(age),max(age),0.01).reshape(-1,1)
plt.plot(x,regression_tree.predict(x),color='blue')
plt.xlabel('Age')
plt.ylabel('Rate Of Approval')
plt.title('Decision Tree Model')
plt.show()
'''





