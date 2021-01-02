# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



data=pd.read_csv("edu_data.csv")
data_c=data.copy()


data_c=data_c.drop(['RecordID'],1)


data_c.Education=data_c['Education'].str.replace('Higher','3')
data_c.Education=data_c['Education'].str.replace('Secondary','2')
data_c.Education=data_c['Education'].str.replace('Primary','1')
data_c.Education=data_c['Education'].str.replace('No education','0')


data_c['Value']=data_c['Value'].fillna(value='0.0',inplace=True)


data_c.Gender=data_c['Gender'].str.replace('F','1')
data_c.Gender=data_c['Gender'].str.replace('M','0')



country=data_c['Country'].unique()

for x in range(70):
    
    data_c.Country=data_c['Country'].str.replace(country[x],str(x))
    
    
data_c.Country=data_c['Country'].str.replace('49ia','50')
data_c.Country=data_c['Country'].str.replace('15 Democratic Republic','16')

data_c['Question Type']=data_c['Question Type'].apply(str)

education0=data_c[data_c['Education'].str.contains("0")]
education1=data_c[data_c['Education'].str.contains("1")] # find the lines for Ages 
education2=data_c[data_c['Education'].str.contains("2")] #merge for the age category
education3=data_c[data_c['Education'].str.contains("3")]   

education0=education0.drop(['Education'],1)
education1=education1.drop(['Education'],1)#now we can drop the age column because i apart the age column multiple dataset
education2=education2.drop(['Education'],1)
education3=education3.drop(['Education'],1)


edu0_f=education0[education0['Gender'].str.contains("1")] #we should part pf two categor 
edu0_m=education0[education0['Gender'].str.contains("0")] #

edu1_f=education1[education1['Gender'].str.contains("1")]
edu1_m=education1[education1['Gender'].str.contains("0")]

edu2_f=education2[education2['Gender'].str.contains("1")]
edu2_m=education2[education2['Gender'].str.contains("0")]


edu3_f=education3[education3['Gender'].str.contains("1")]
edu3_m=education3[education3['Gender'].str.contains("0")]



edu0_m_avg=[]
edu0_f_avg=[]
edu1_m_avg=[]
edu1_f_avg=[]
edu2_m_avg=[]
edu2_f_avg=[]
edu3_f_avg=[]
edu3_m_avg=[]


for x in range(70):
    a=edu0_m['Value'][edu0_m['Country'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu0_m_avg.append(a)

for x in range(70):
    a=edu0_f['Value'][edu0_f['Country'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu0_f_avg.append(a)
    
for x in range(70):
    a=edu1_m['Value'][edu1_m['Country'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu1_m_avg.append(a)

for x in range(70):
    a=edu1_f['Value'][edu1_f['Country'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu1_f_avg.append(a)
    
for x in range(70):
    a=edu2_m['Value'][edu2_m['Country'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu2_m_avg.append(a)

for x in range(70):
    a=edu2_f['Value'][edu2_f['Country'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu2_f_avg.append(a)

for x in range(70):
    a=edu3_m['Value'][edu3_m['Country'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu3_m_avg.append(a)

for x in range(70):
    a=edu3_f['Value'][edu3_f['Country'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu3_f_avg.append(a)



edu0=[]                                   #merging  two male and female average value each other 
edu1=[]
edu2=[]
edu3=[]
for x in range(70):                  
    edu0.append(edu0_m_avg[x])
    edu0.append(edu0_f_avg[x])
    edu1.append(edu1_m_avg[x])
    edu1.append(edu1_f_avg[x])
    edu2.append(edu2_m_avg[x])
    edu2.append(edu2_f_avg[x])
    edu3.append(edu3_m_avg[x])
    edu3.append(edu3_f_avg[x])

country=np.arange(70) #######we should have 3 different type column arrange the columns
country_list=[]

for x in range(70):         #country column should be like 0,0,1,1,2,2......68,68,69,69
    country_list.append(country[x])
    country_list.append(country[x])

gender=[0,1]

gender_list=[] #create a gender column like that 0,1,0,1,0,1
for x in range(70):
    for y in range(2):
        gender_list.append(gender[y])
        


df0 = pd.DataFrame(country_list, columns=["Country"])  #Creating new DataFrame for Education==0
df0['Value']=edu0
df0['Gender'] = gender_list



df1 = pd.DataFrame(country_list, columns=["Country"])  #Creating new DataFrame for Education==1 
df1['Value']=edu1
df1['Gender'] = gender_list

df2 = pd.DataFrame(country_list, columns=["Country"])  #Creating new DataFrame for Education==2
df2['Value']=edu2
df2['Gender'] = gender_list



df3 = pd.DataFrame(country_list, columns=["Country"])  #Creating new DataFrame for Education==3
df3['Value']=edu3
df3['Gender'] = gender_list


'''

v=df0.values

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, init='k-means++',random_state=0)

# Kümeleme işlemi yap
km.fit(v)

# Tahmin işlemi yapıyoruz.
predict = km.predict(v)

import matplotlib.pyplot as plt
plt.scatter(v[predict==0,0],v[predict==0,1],s=50,color='red')
plt.scatter(v[predict==1,0],v[predict==1,1],s=50,color='blue')
plt.scatter(v[predict==2,0],v[predict==2,1],s=50,color='green')
plt.title('K-Means Clustering')
plt.xlabel("Countries")
plt.ylabel("Average Values")
plt.show()
'''

















v=df0.iloc[:,0:2]
gender=df0.Gender.values





from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(v,gender,test_size=0.33,random_state=0)


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0)

# Makinemizi eğittik
lr.fit(x_train,y_train)

# Test veri kümemizi verdik ve cinsiyet tahmin etmesini sağladık
result = lr.predict(x_test)

count=0
# Sonuçları ekrana yazdırdık
for i in range(len(result)):
    #print("Tahmin : " + str(result[i]) + ", Gerçek Değer : " + str(gender[i]))
    if result[i]==gender[i]:
        count=count+1

print(str(len(result))+' prediction'+'==> '+str(count)+' right prediction.')









v=df1.iloc[:,0:2]
gender=df1.Gender.values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(v,gender,test_size=0.33,random_state=0)



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0)

# Makinemizi eğittik
lr.fit(x_train,y_train)

# Test veri kümemizi verdik ve cinsiyet tahmin etmesini sağladık
result = lr.predict(x_test)

count=0
# Sonuçları ekrana yazdırdık
for i in range(len(result)):
    #print("Tahmin : " + str(result[i]) + ", Gerçek Değer : " + str(gender[i]))
    if result[i]==gender[i]:
        count=count+1
print(str(len(result))+' prediction'+'==> '+str(count)+' right prediction.')



v=df2.iloc[:,0:2]
gender=df2.Gender.values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(v,gender,test_size=0.33,random_state=0)



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0)

# Makinemizi eğittik
lr.fit(x_train,y_train)

# Test veri kümemizi verdik ve cinsiyet tahmin etmesini sağladık
result = lr.predict(x_test)

count=0
# Sonuçları ekrana yazdırdık
for i in range(len(result)):
    #print("Tahmin : " + str(result[i]) + ", Gerçek Değer : " + str(gender[i]))
    if result[i]==gender[i]:
        count=count+1
print(str(len(result))+' prediction'+'==> '+str(count)+' right prediction.')





v=df3.iloc[:,0:2]
gender=df3.Gender.values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(v,gender,test_size=0.33,random_state=0)



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0)

# Makinemizi eğittik
lr.fit(x_train,y_train)

# Test veri kümemizi verdik ve cinsiyet tahmin etmesini sağladık
result = lr.predict(x_test)

count=0
# Sonuçları ekrana yazdırdık
for i in range(len(result)):
    #print("Tahmin : " + str(result[i]) + ", Gerçek Değer : " + str(gender[i]))
    if result[i]==gender[i]:
        count=count+1
print(str(len(result))+' prediction'+'==> '+str(count)+' right prediction.')




edu3=education3[education3['Gender'].str.contains("0")]

edu0=education0.drop(['Gender'],1)
edu1=education1.drop(['Gender'],1)
edu2=education2.drop(['Gender'],1)
edu3=education3.drop(['Gender'],1)

edu0_val=[]
edu1_val=[]
edu2_val=[]
edu3_val=[]


for x in range(70):
    a=edu0['Value'][edu0['Country'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu0_val.append(a)

for x in range(70):
    a=edu1['Value'][edu1['Country'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu1_val.append(a)
for x in range(70):
    a=edu2['Value'][edu2['Country'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu2_val.append(a)
for x in range(70):
    a=edu3['Value'][edu3['Country'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu3_val.append(a)



country=np.arange(70)

df0 = pd.DataFrame(country, columns=["Country"])  #Creating new DataFrame for Education==0
df0['Value']=edu0_val

df1 = pd.DataFrame(country, columns=["Country"])  #Creating new DataFrame for Education==1
df1['Value']=edu1_val

df2 = pd.DataFrame(country, columns=["Country"])  #Creating new DataFrame for Education==2
df2['Value']=edu2_val

df3 = pd.DataFrame(country, columns=["Country"])  #Creating new DataFrame for Education==3
df3['Value']=edu3_val







###Compare the regressions

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# LinearRegression sınıfından bir nesne ürettik
lr = LinearRegression()

# PolynomialFeatures sınıfından bir nesne ürettik.
# Buradaki degree parametresi polinomun derecesidir.
# degree kısmına 1 verirsek doğrusal bir doğru çizilecektir.
# degree ne kadar arttırılırsa o kadar sağlıklı bir sonuç elde edebiliriz.
poly = PolynomialFeatures(degree=4)

# Makineyi eğitmeden önce egitim kolonundaki değerleri PolynomialFeatures ile dönüşüm yapıyoruz.

country0=df0.Country.values.reshape(-1,1)
value0=df0.Value.values.reshape(-1,1)
value1=df1.Value.values.reshape(-1,1)
value2=df2.Value.values.reshape(-1,1)
value3=df3.Value.values.reshape(-1,1)





poly_new = poly.fit_transform(country0)



# Makineyi eğitiyoruz.
lr.fit(poly_new, value0)
predict0 = lr.predict(poly_new)

lr.fit(poly_new,value1)
predict1=lr.predict(poly_new)

lr.fit(poly_new,value2)
predict2=lr.predict(poly_new)

lr.fit(poly_new,value3)
predict3=lr.predict(poly_new)


c=np.arange(70).reshape(-1,1)
# Grafik şeklinde ekrana basmak için
#plt.scatter(df0.Country.values, df0.Value.values, color='red')
plt.plot(df0.Country.values, predict0, color='blue')
plt.plot(df1.Country.values, predict1, color='red')
plt.plot(df2.Country.values, predict2, color='green')
plt.plot(df3.Country.values, predict3, color='orange')
plt.title("Compare the Education Level by Polynomial Regression")
plt.xlabel("Countries")
plt.ylabel("Rate of Approval")
plt.show()


















'''
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, result)

print(accuracy)

from sklearn.metrics import confusion_matrix
# Parametre olarak karşılaştıracağımız verileri giriyoruz.
# y_test : Cinsiyet Test Verisi
# result : x_test verisinden tahmin ettiğimiz cinsiyet verileri
cm = confusion_matrix(y_test,result)
print(cm)
'''




