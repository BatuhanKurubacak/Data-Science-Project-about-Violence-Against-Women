# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor


data=pd.read_csv("age_data_value.csv")

data_c=data.copy()

print(data_c.columns)

data_c=data_c.drop(['Question Type'],1)
data_c=data_c.drop(['RecordID'],1)

data_c.Age=data_c['Age'].str.replace('15-24','1')
data_c.Age=data_c['Age'].str.replace('25-34','2')
data_c.Age=data_c['Age'].str.replace('35-49','3')

data_c.Gender=data_c['Gender'].str.replace('F','1')
data_c.Gender=data_c['Gender'].str.replace('M','0')

#data_c.Gender=data_c.Gender.apply(int)
#data_c.Age= data_c.Age.apply(int)

country=data_c['Country'].unique()


print(data_c['Country'].unique())

for x in range(70):
    
    data_c.Country=data_c['Country'].str.replace(country[x],str(x))


data_c.Country=data_c['Country'].str.replace('49ia','50')
data_c.Country=data_c['Country'].str.replace('15 Democratic Republic','16')
    


age1=data_c[data_c['Age'].str.contains("1")] # find the lines for Ages 
age2=data_c[data_c['Age'].str.contains("2")] #merge for the age category
age3=data_c[data_c['Age'].str.contains("3")]


age1=age1.drop(['Age'],1)#now we can drop the age column because i apart the age column multiple dataset
age2=age2.drop(['Age'],1)
age3=age3.drop(['Age'],1)


print('length of Age1: '+str(len(age1)))#So as you can see all categories length is equal.
print('length of Age2: '+str(len(age2)))
print('length of Age3: '+str(len(age3)))

age1_female=age1[age1['Gender'].str.contains("1")] #we should part pf two categor 
age1_male=age1[age1['Gender'].str.contains("0")] #

age2_female=age2[age2['Gender'].str.contains("1")] # 
age2_male=age2[age2['Gender'].str.contains("0")] #


age3_female=age3[age3['Gender'].str.contains("1")] # 
age3_male=age3[age3['Gender'].str.contains("0")] #

age1_male=age1_male.drop(['Gender'],1)#now we can drop the gender column because i apart the gender column multiple dataset
age1_female=age1_female.drop(['Gender'],1)
age2_male=age2_male.drop(['Gender'],1)
age2_female=age2_female.drop(['Gender'],1)
age3_male=age3_male.drop(['Gender'],1)
age3_female=age3_female.drop(['Gender'],1)


print('length of Age1 and Women: '+str(len(age1_female)))#So as you can see all categories length is equal.
print('length of Age1 and Men: '+str(len(age1_male)))
print('length of Age2 and Women: '+str(len(age2_female)))
print('length of Age2 and Men: '+str(len(age2_male)))
print('length of Age3 and Women: '+str(len(age3_female)))
print('length of Age3 and Men: '+str(len(age3_male)))

average_list=[]

      #we should take a average value of questions

data_c.Country=data_c['Country'].str.replace(country[x],str(x))


for x in range(0,420,6):

    val1=age3_male.iloc[x:x+6,0:2]
    #print(val1)
    val1['Value'].apply(float)
    s1=val1['Value'].sum()
    s1=s1/6
    s1=format(round(s1,2))
    average_list.append(s1)
    

city_list=[]

for x in range(0,70):
    city_list.append(x)

df = pd.DataFrame(city_list, columns=["Country"]) 
df['Average_Value'] = average_list
df=df.sort_values(by=['Average_Value'])



country=df.Country.values.reshape(-1,1)
val=df.Average_Value.values.reshape(-1,1)

print(val)

regression=LinearRegression()
regression=regression.fit(country,val)


df=df.astype(float)

x=np.arange(min(df.Country),max(df.Country)).reshape(-1,1)
plt.plot(x,regression.predict(x),color='red')
plt.scatter(df.Country,df.Average_Value)
plt.xlabel("Countries")
plt.ylabel("Rate of Approval")
plt.title("Linear Regression")
plt.show()

#print(r2_score(val,regression.predict(country)))
'''
regression_tree=DecisionTreeRegressor() # i used decision tree regressor but it's not make sense
regression_tree.fit(country,val)
print(r2_score(val,regression_tree.predict(country)))


plt.scatter(df.Country,df.Average_Value,color='red')
x=np.arange(min(country),max(country),0.01).reshape(-1,1)
plt.plot(x,regression_tree.predict(x),color='blue')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Decision Tree Model')
plt.show()
'''
'''
for x in range(10,70):                                               #alternative calculate average rate of approval but there is a mistake 
    country1=age3_male[age3_male['Country'].str.contains(str(x))]
    values=country1.iloc[:,1]
    #print(type(values))
    values.apply(float)
    s=values.sum()
    s=s/6
    s=format(round(s, 2))
    average_list.append(s)
''' 