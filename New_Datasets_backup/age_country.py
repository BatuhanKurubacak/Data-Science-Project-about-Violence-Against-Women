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

unique_country=data_c['Country'].unique()
print(data_c['Country'].unique())

country_list=data_c['Country'].unique()
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


#print('length of Age1: '+str(len(age1)))#So as you can see all categories length is equal.
#print('length of Age2: '+str(len(age2)))
#print('length of Age3: '+str(len(age3)))

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


#print('length of Age1 and Women: '+str(len(age1_female)))#So as you can see all categories length is equal.
#print('length of Age1 and Men: '+str(len(age1_male)))
#print('length of Age2 and Women: '+str(len(age2_female)))
#print('length of Age2 and Men: '+str(len(age2_male)))
#print('length of Age3 and Women: '+str(len(age3_female)))
#print('length of Age3 and Men: '+str(len(age3_male)))

average_list=[]

      #we should take a average value of questions

#data_c.Country=data_c['Country'].str.replace(country[x],str(x))


for x in range(0,420,6): #we have 6 different type question

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
#df=df.sort_values(by=['Average_Value'])




country=df.Country.values.reshape(-1,1)
val=df.Average_Value.values.reshape(-1,1)

#print(val)
regression1=LinearRegression()
regression1=regression1.fit(country,val)

df=df.astype(float)
age3_m_avg=df

x=np.arange(min(df.Country),max(df.Country)).reshape(-1,1)
plt.plot(x,regression1.predict(x),color='red')
plt.scatter(df.Country,df.Average_Value)
plt.xlabel("Countries")
plt.ylabel("Rate of Approval")
plt.title("Linear Regression for Man Aged 35-49")
plt.show()



average_list1=[]

      #we should take a average value of questions


for x in range(0,420,6): #we have 6 different type question

    val2=age3_female.iloc[x:x+6,0:2]
    #print(val1)
    val2['Value'].apply(float)
    s2=val2['Value'].sum()
    s2=s2/6
    s2=format(round(s2,2))
    average_list1.append(s2)
    

city_list1=[]

for x in range(0,70):
    city_list1.append(x)

df1 = pd.DataFrame(city_list1, columns=["Country"]) 
df1['Average_Value'] = average_list1
#df1=df.sort_values(by=['Average_Value'])



country1=df1.Country.values.reshape(-1,1)
val1=df1.Average_Value.values.reshape(-1,1)

#print(val)

regression2=LinearRegression()
regression2=regression2.fit(country1,val1)


df1=df1.astype(float)
age3_f_avg=df1


x=np.arange(min(df1.Country),max(df1.Country)).reshape(-1,1)
plt.plot(x,regression2.predict(x),color='red')
plt.scatter(df1.Country,df1.Average_Value)
plt.xlabel("Countries1")
plt.ylabel("Rate of Approval")
plt.title("Linear Regression for Women Aged 35-49")
plt.show()



average_list1=[]

      #we should take a average value of questions


for x in range(0,420,6): #we have 6 different type question

    val2=age2_female.iloc[x:x+6,0:2]
    #print(val1)
    val2['Value'].apply(float)
    s2=val2['Value'].sum()
    s2=s2/6
    s2=format(round(s2,2))
    average_list1.append(s2)
    

city_list1=[]

for x in range(0,70):
    city_list1.append(x)

df1 = pd.DataFrame(city_list1, columns=["Country"]) 
df1['Average_Value'] = average_list1
#df1=df.sort_values(by=['Average_Value'])


country1=df1.Country.values.reshape(-1,1)
val1=df1.Average_Value.values.reshape(-1,1)

#print(val)

regression3=LinearRegression()
regression3=regression3.fit(country1,val1)


df1=df1.astype(float)
age2_f_avg=df1
x=np.arange(min(df1.Country),max(df1.Country)).reshape(-1,1)
plt.plot(x,regression3.predict(x),color='red')
plt.scatter(df1.Country,df1.Average_Value)
plt.xlabel("Countries")
plt.ylabel("Rate of Approval")
plt.title("Linear Regression for Women Aged 25-34")
plt.show()




average_list1=[]     
#we should take a average value of questions

for x in range(0,420,6): #we have 6 different type question

    val2=age2_male.iloc[x:x+6,0:2]
    #print(val1)
    val2['Value'].apply(float)
    s2=val2['Value'].sum()
    s2=s2/6
    s2=format(round(s2,2))
    average_list1.append(s2)
    

city_list1=[]

for x in range(0,70):
    city_list1.append(x)

df1 = pd.DataFrame(city_list1, columns=["Country"]) 
df1['Average_Value'] = average_list1
#df1=df.sort_values(by=['Average_Value'])



country1=df1.Country.values.reshape(-1,1)
val1=df1.Average_Value.values.reshape(-1,1)

#print(val)

regression4=LinearRegression()
regression4=regression4.fit(country1,val1)


df1=df1.astype(float)
age2_m_avg=df1

x=np.arange(min(df1.Country),max(df1.Country)).reshape(-1,1)
plt.plot(x,regression4.predict(x),color='red')
plt.scatter(df1.Country,df1.Average_Value)
plt.xlabel("Countries")
plt.ylabel("Rate of Approval")
plt.title("Linear Regression for Men Aged 25-34")
plt.show()


average_list1=[]     
#we should take a average value of questions

for x in range(0,420,6): #we have 6 different type question

    val2=age1_male.iloc[x:x+6,0:2]
    #print(val1)
    val2['Value'].apply(float)
    s2=val2['Value'].sum()
    s2=s2/6
    s2=format(round(s2,2))
    average_list1.append(s2)
    

city_list1=[]

for x in range(0,70):
    city_list1.append(x)

df1 = pd.DataFrame(city_list1, columns=["Country"]) 
df1['Average_Value'] = average_list1
#df1=df.sort_values(by=['Average_Value'])



country1=df1.Country.values.reshape(-1,1)
val1=df1.Average_Value.values.reshape(-1,1)

#print(val)

regression5=LinearRegression()
regression5=regression5.fit(country1,val1)


df1=df1.astype(float)

age1_m_avg=df1


x=np.arange(min(df1.Country),max(df1.Country)).reshape(-1,1)
plt.plot(x,regression5.predict(x),color='red')
plt.scatter(df1.Country,df1.Average_Value)
plt.xlabel("Countries")
plt.ylabel("Rate of Approval")
plt.title("Linear Regression for Men Aged 15-24")
plt.show()


average_list1=[]     
#we should take a average value of questions

for x in range(0,420,6): #we have 6 different type question

    val2=age1_female.iloc[x:x+6,0:2]
    #print(val1)
    val2['Value'].apply(float)
    s2=val2['Value'].sum()
    s2=s2/6
    s2=format(round(s2,2))
    average_list1.append(s2)
    

city_list1=[]

for x in range(0,70):
    city_list1.append(x)

df1 = pd.DataFrame(city_list1, columns=["Country"]) 
df1['Average_Value'] = average_list1
#df1=df.sort_values(by=['Average_Value'])



country1=df1.Country.values.reshape(-1,1)
val1=df1.Average_Value.values.reshape(-1,1)

#print(val)

regression6=LinearRegression()
regression6=regression6.fit(country1,val1)


df1=df1.astype(float)

x=np.arange(min(df1.Country),max(df1.Country)).reshape(-1,1)
plt.plot(x,regression6.predict(x),color='red')
plt.scatter(df1.Country,df1.Average_Value)
plt.xlabel("Countries")
plt.ylabel("Rate of Approval")
plt.title("Linear Regression for Women Aged 15-24")
plt.show()

age1_f_avg=df1



z=np.arange(min(df1.Country),max(df1.Country)).reshape(-1,1)
plt.plot(z,regression6.predict(z),color='red')#women 15-24
plt.plot(z,regression5.predict(z),color='blue')#men 15-24
plt.plot(z,regression4.predict(z),color='green')#men 25-39
plt.plot(z,regression3.predict(z),color='orange')#women 25-39
plt.plot(z,regression2.predict(z),color='black')#women 40-54
plt.plot(z,regression1.predict(z),color='yellow')#men 40-54
plt.xlabel("Countries")
plt.ylabel("Rate of Approval")
plt.title("Compare the Linear Regression by Categories ")
plt.show()


#age1_f_avg -- FEMALE AGED 15-24 , VALUE COLUMN IS AVERAGE VALUE OF THE SIX TYPE OF QUESTION
#age1_m_avg -- MALE AGED 15-24 , VALUE COLUMN IS AVERAGE VALUE OF THE SIX TYPE OF QUESTION
#age2_f_avg -- FEMALE AGED 25-34 , VALUE COLUMN IS AVERAGE VALUE OF THE SIX TYPE OF QUESTION
#age2_m_avg -- MALE AGED 25-34 , VALUE COLUMN IS AVERAGE VALUE OF THE SIX TYPE OF QUESTION
#age3_f_avg -- FEMALE AGED 35-49 , VALUE COLUMN IS AVERAGE VALUE OF THE SIX TYPE OF QUESTION
#age3_m_avg -- MALE AGED 35-49 , VALUE COLUMN IS AVERAGE VALUE OF THE SIX TYPE OF QUESTION


age1_f_avg=age1_f_avg.sort_values(by=['Average_Value'])
age1_m_avg=age1_m_avg.sort_values(by=['Average_Value'])
age2_f_avg=age2_f_avg.sort_values(by=['Average_Value'])
age2_m_avg=age2_m_avg.sort_values(by=['Average_Value'])
age3_f_avg=age3_f_avg.sort_values(by=['Average_Value'])
age3_m_avg=age3_m_avg.sort_values(by=['Average_Value'])






fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
avg=age1_f_avg.Average_Value
country_list1=age1_f_avg.Country
ax.bar(country_list1,avg,edgecolor='black')
plt.title('Approval Rate of Women aged 15-24 by Country')
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
avg=age1_m_avg.Average_Value
country_list1=age1_m_avg.Country
ax.bar(country_list1,avg,color='green', edgecolor='black')
plt.title('Approval Rate of Men aged 15-24 by Country')
plt.show()


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
avg=age2_m_avg.Average_Value
country_list1=age2_m_avg.Country
ax.bar(country_list1,avg,color='red',edgecolor='black')
plt.title('Approval Rate of Men aged 25-34 by Country')
plt.show()



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
avg=age2_f_avg.Average_Value
country_list1=age2_f_avg.Country
ax.bar(country_list1,avg,color='orange', edgecolor='black')
plt.title('Approval Rate of Women aged 25-34 by Country')
plt.show()


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
avg=age3_f_avg.Average_Value
country_list1=age3_f_avg.Country
ax.bar(country_list1,avg,color='yellow',edgecolor='black')
plt.title('Approval Rate of Women aged 35-49 by Country')
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
avg=age3_m_avg.Average_Value
country_list1=age3_m_avg.Country
ax.bar(country_list1,avg,color='purple',edgecolor='black')
plt.title('Approval Rate of Men aged 35-49 by Country')
plt.show()

#print(country_list[65])




def country(x): #I should the learn which number represented by countries
    
    answer=country_list[x]
    return answer



high=age1_f_avg.iloc[-1,:]
print(str(country(int(high.Country)))+'has the highest rate of approval : '+str(high.Average_Value)+" Women Aged 15-24")
low=age1_f_avg.iloc[0,:]
print(str(country(int(low.Country)))+'has the lowest rate of approval : '+str(low.Average_Value)+" Women Aged 15-24"+'\n')


high=age1_m_avg.iloc[-1,:]
print(str(country(int(high.Country)))+'has the highest rate of approval : '+str(high.Average_Value)+" Men Aged 15-24")
low=age1_m_avg.iloc[12,:]
print(str(country(int(low.Country)))+'has the lowest rate of approval : '+str(low.Average_Value)+" Men Aged 15-24"+'\n')



high=age2_m_avg.iloc[-1,:]
print(str(country(int(high.Country)))+'has the highest rate of approval : '+str(high.Average_Value)+" Men Aged 25-34")
low=age2_m_avg.iloc[12,:]
print(str(country(int(low.Country)))+'has the lowest rate of approval : '+str(low.Average_Value)+" Men Aged 25-34"+'\n')


high=age2_f_avg.iloc[-1,:]
print(str(country(int(high.Country)))+'has the highest rate of approval : '+str(high.Average_Value)+" Women Aged 25-34")
low=age2_f_avg.iloc[0,:]
print(str(country(int(low.Country)))+'has the lowest rate of approval : '+str(low.Average_Value)+" Women Aged 25-34"+'\n')

high=age3_f_avg.iloc[-1,:]
print(str(country(int(high.Country)))+'has the highest rate of approval : '+str(high.Average_Value)+" Women Aged 35-49")
low=age3_f_avg.iloc[0,:]
print(str(country(int(low.Country)))+'has the lowest rate of approval : '+str(low.Average_Value)+" Women Aged 35-49"+"\n")


high=age3_m_avg.iloc[-1,:]
print(str(country(int(high.Country)))+'has the highest rate of approval :'+str(high.Average_Value)+" Men Aged 35-49")
low=age3_m_avg.iloc[12,:]
print(str(country(int(low.Country)))+'has the lowest rate of approval : '+str(low.Average_Value)+" Men Aged 35-49")










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
for x in range(10,70):        #alternative calculate average rate of approval but there is a mistake 
    country1=age3_male[age3_male['Country'].str.contains(str(x))]
    values=country1.iloc[:,1]
    #print(type(values))
    values.apply(float)
    s=values.sum()
    s=s/6
    s=format(round(s, 2))
    average_list.append(s)
''' 
