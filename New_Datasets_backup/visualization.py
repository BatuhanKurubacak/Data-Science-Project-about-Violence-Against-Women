# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

for x in range(70):       #we should take a average value of questions

    country1=age3_male[age3_male['Country']].str.contains(str(x))
    values=country1.Value.values()
    s=0
    for x in values:
        s=s+x
    average=s/6
    average_list.append(average)

