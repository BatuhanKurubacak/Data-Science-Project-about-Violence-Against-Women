# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



data=pd.read_csv("age_data_value.csv")

data_c=data.copy()



data_c=data_c.drop(['RecordID'],1)
data_c=data_c.drop(['Country'],1)
data_c=data_c.drop(['Gender'],1)

print(data_c.columns)

data_c.Age=data_c['Age'].str.replace('15-24','1')
data_c.Age=data_c['Age'].str.replace('25-34','2')
data_c.Age=data_c['Age'].str.replace('35-49','3')
data_c['Question Type']=data_c['Question Type'].apply(str)

age3=data_c[data_c['Age'].str.contains("3")]
age2=data_c[data_c['Age'].str.contains("2")]
age1=data_c[data_c['Age'].str.contains("1")]


age3=age3.drop(['Age'],1)
age2=age2.drop(['Age'],1)
age1=age1.drop(['Age'],1)


age1_avg=[]

for x in range(1,7):
    a=age1['Value'][age1['Question Type'].str.contains(str(x))].mean()
    a=format(round(a,2))
    age1_avg.append(a)

age2_avg=[]

for x in range(1,7):
    a=age2['Value'][age2['Question Type'].str.contains(str(x))].mean()
    a=format(round(a,2))
    age2_avg.append(a)

age3_avg=[]

for x in range(1,7):
    a=age3['Value'][age3['Question Type'].str.contains(str(x))].mean()
    a=format(round(a,2))
    age3_avg.append(a)


questions=range(1,7)


df = pd.DataFrame(age1_avg, columns=["15-24"])
df['25-34'] = age2_avg
df['35-49'] = age3_avg
df['Questions']=questions



df['15-24']=df['15-24'].apply(float)
df['25-34']=df['25-34'].apply(float)
df['35-49']=df['35-49'].apply(float)


top_medians = df.sort_values("15-24")
top_medians.plot(x="Questions", y=["15-24", "25-34", "35-49"], kind="bar")
plt.title("Approval Rate of Questions by Age Categories")
plt.show()

plt.scatter(df.Questions,df['15-24'])
plt.show()



x=df['Questions'].values.reshape(-1,1)
y=df['15-24'].values.reshape(-1,1)

y1=df['25-34'].values.reshape(-1,1)
y2=df['35-49'].values.reshape(-1,1)




lr = LinearRegression()

# PolynomialFeatures sınıfından bir nesne ürettik.
# Buradaki degree parametresi polinomun derecesidir.
# degree kısmına 1 verirsek doğrusal bir doğru çizilecektir.
# degree ne kadar arttırılırsa o kadar sağlıklı bir sonuç elde edebiliriz.
poly = PolynomialFeatures(degree=4)

# Makineyi eğitmeden önce soru kolonundaki değerleri PolynomialFeatures ile dönüşüm yapıyoruz.
soru_poly = poly.fit_transform(x)

# Makineyi eğitiyoruz.
lr.fit(soru_poly, y)

# Makineyi eğittikten sonra bir tahmin yaptırtıyoruz.
predict = lr.predict(soru_poly)

plt.scatter(x, y, color='red')
plt.plot(x, predict, color='blue')
plt.title("Polynomial Regression by Aged 15-24")
plt.show()


lr.fit(soru_poly, y1)
predict1 = lr.predict(soru_poly)

plt.scatter(x, y1, color='green')
plt.plot(x, predict1, color='blue')
plt.title("Polynomial Regression by Aged 25-34")
plt.show()


lr.fit(soru_poly, y2)
predict2 = lr.predict(soru_poly)

plt.scatter(x, y, color='black')
plt.plot(x, predict2, color='blue')
plt.title("Polynomial Regression by Aged 35-49")
plt.show()


plt.plot(x, predict2, color='black')
plt.plot(x, predict1, color='green')
plt.plot(x, predict, color='blue')
plt.title("Comparison of Polynomial Regressions for Ages")
plt.show()