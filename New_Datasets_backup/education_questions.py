# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import AgglomerativeClustering


data=pd.read_csv("edu_data.csv")
data_c=data.copy()


data_c=data_c.drop(['RecordID'],1)
data_c=data_c.drop(['Country'],1)

data_c.Education=data_c['Education'].str.replace('Higher','3')
data_c.Education=data_c['Education'].str.replace('Secondary','2')
data_c.Education=data_c['Education'].str.replace('Primary','1')
data_c.Education=data_c['Education'].str.replace('No education','0')


data_c.Gender=data_c['Gender'].str.replace('F','1')
data_c.Gender=data_c['Gender'].str.replace('M','0')

data_c['Question Type']=data_c['Question Type'].apply(str)


    
edu0=data_c[data_c['Education'].str.contains("0")]
edu1=data_c[data_c['Education'].str.contains("1")]
edu2=data_c[data_c['Education'].str.contains("2")]
edu3=data_c[data_c['Education'].str.contains("3")]



edu0=edu0.drop(['Education'],1)
edu1=edu1.drop(['Education'],1)
edu2=edu2.drop(['Education'],1)
edu3=edu3.drop(['Education'],1)



edu0_m=edu0[edu0['Gender'].str.contains("0")]
edu0_f=edu0[edu0['Gender'].str.contains("1")]

edu1_m=edu1[edu1['Gender'].str.contains("0")]
edu1_f=edu1[edu1['Gender'].str.contains("1")]


edu2_m=edu2[edu2['Gender'].str.contains("0")]
edu2_f=edu2[edu2['Gender'].str.contains("1")]

edu3_m=edu3[edu3['Gender'].str.contains("0")]
edu3_f=edu3[edu3['Gender'].str.contains("1")]

'''
data_c['Value']=data_c['Value'].apply(str)
edu_null=data_c[data_c['Value'].str.contains("0.0")]
edu_null0=edu_null[edu_null['Gender'].str.contains("0")]
edu_null1=edu_null[edu_null['Gender'].str.contains("1")]
'''


edu0_m_avg=[]
edu0_f_avg=[]
edu1_m_avg=[]
edu1_f_avg=[]
edu2_m_avg=[]
edu2_f_avg=[]
edu3_m_avg=[]
edu3_f_avg=[]




for x in range(1,7):
    a=edu0_m['Value'][edu0_m['Question Type'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu0_m_avg.append(a)

for x in range(1,7):
    a=edu0_f['Value'][edu0_f['Question Type'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu0_f_avg.append(a)
    
for x in range(1,7):
    a=edu1_m['Value'][edu1_m['Question Type'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu1_m_avg.append(a)

for x in range(1,7):
    a=edu1_f['Value'][edu1_f['Question Type'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu1_f_avg.append(a)
    
for x in range(1,7):
    a=edu2_m['Value'][edu2_m['Question Type'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu2_m_avg.append(a)

for x in range(1,7):
    a=edu2_f['Value'][edu2_f['Question Type'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu2_f_avg.append(a)

for x in range(1,7):
    a=edu3_m['Value'][edu3_m['Question Type'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu3_m_avg.append(a)

for x in range(1,7):
    a=edu3_f['Value'][edu3_f['Question Type'].str.contains(str(x))].mean()
    a=format(round(a,2))
    edu3_f_avg.append(a)


questions=np.arange(1,7) 


df0 = pd.DataFrame(questions, columns=["Question"])
df0['Male_Approval']=edu0_m_avg
df0['Female_Approval']=edu0_f_avg
df0['Male_Approval']=df0['Male_Approval'].apply(float)
df0['Female_Approval']=df0['Female_Approval'].apply(float)


df1 = pd.DataFrame(questions, columns=["Question"])
df1['Male_Approval']=edu1_m_avg
df1['Female_Approval']=edu1_f_avg
df1['Male_Approval']=df1['Male_Approval'].apply(float)
df1['Female_Approval']=df1['Female_Approval'].apply(float)

df2 = pd.DataFrame(questions, columns=["Question"])
df2['Male_Approval']=edu2_m_avg
df2['Female_Approval']=edu2_f_avg
df2['Male_Approval']=df2['Male_Approval'].apply(float)
df2['Female_Approval']=df2['Female_Approval'].apply(float)

df3 = pd.DataFrame(questions, columns=["Question"])
df3['Male_Approval']=edu3_m_avg
df3['Female_Approval']=edu3_f_avg
df3['Male_Approval']=df3['Male_Approval'].apply(float)
df3['Female_Approval']=df3['Female_Approval'].apply(float)



top_medians = df0.sort_values("Male_Approval")
top_medians.plot(x="Question", y=["Male_Approval", "Female_Approval"], kind="bar")
plt.title("Approval Rate of Questions for No Education by Gender Categories")
plt.show()

top_medians = df1.sort_values("Male_Approval")
top_medians.plot(x="Question", y=["Male_Approval", "Female_Approval"], kind="bar")
plt.title("Approval Rate of Questions for Primary by Gender Categories")
plt.show()

top_medians = df2.sort_values("Male_Approval")
top_medians.plot(x="Question", y=["Male_Approval", "Female_Approval"], kind="bar")
plt.title("Approval Rate of Questions for Secondary by Gender Categories")
plt.show()

top_medians = df3.sort_values("Male_Approval")
top_medians.plot(x="Question", y=["Male_Approval", "Female_Approval"], kind="bar")
plt.title("Approval Rate of Questions for Higher by Gender Categories")
plt.show()





v = df0.values
ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean',linkage='complete')

# Kümeleme ve tahmin işlemi yap
predict = ac.fit_predict(v)

# Dendogram grafiği gösterimi
import scipy.cluster.hierarchy as sch



# Grafik şeklinde ekrana basmak için
plt.scatter(v[predict==0,0],v[predict==0,1],s=50,color='red')
plt.scatter(v[predict==1,0],v[predict==1,1],s=50,color='blue')
plt.scatter(v[predict==2,0],v[predict==2,1],s=50,color='green')
plt.xlabel('Question Numbers')
plt.ylabel('Rate of Approval')
plt.title('Hierarchical Clustering Education Dataset')
plt.show()


# v = verilerimiz
# method = AgglomerativeClustering'in linkage parametresi ile aynı parametreyi veriyoruz. ( 'ward' )
dendrogram = sch.dendrogram(sch.linkage(v,method='complete'))
plt.title('Hierarchical Clustering Education Dataset')
plt.show()




# Bağımlı Değişkeni ( species) bir değişkene atadık
gender = edu0.Gender.values

# Veri kümemizi test ve train şekinde bölüyoruz
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(edu0.iloc[:,1:3],gender,test_size=0.33,random_state=0)


# GaussianNB sınıfını import ettik
# 3 tane farklı Naive Bayes Sınıfı vardır.
# GaussianNB : Tahmin edeceğiniz veri veya kolon sürekli (real,ondalıklı vs.) ise
# BernoulliNB : Tahmin edeceğiniz veri veya kolon ikili ise ( Evet/Hayır , Sigara içiyor/ İçmiyor vs.)
# MultinomialNB : Tahmin edeceğiniz veri veya kolon nominal ise ( Int sayılar )
# Duruma göre bu üç sınıftan birini seçebilirsiniz. Modelin başarı durumunu etkiler.
from sklearn.naive_bayes import BernoulliNB

# GaussianNB sınıfından bir nesne ürettik
gnb = BernoulliNB()

# Makineyi eğitiyoruz
gnb.fit(x_train, y_train.ravel())

# Test veri kümemizi verdik ve gender tahmin etmesini sağladık
result = gnb.predict(x_test)

# Karmaşıklık matrisi
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,result)
print(cm)

# Başarı Oranı
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, result)
# Sonuç : 0.96
print(accuracy)

