# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

data=pd.read_csv("violence_data.csv")

#print(data.head)

c_data=data.copy()
#print(c_data.columns)

c_data = c_data.drop('Survey Year',axis=1)# delete Survey year column
#print(c_data)

#print(type(c_data.Value))
c_data['Value'].fillna(value=0,inplace=True)#fill the  blank spaces in value column
#print(c_data)




value_data=c_data[c_data['Value']>=0][c_data['Value']<=7.0]["Value"]#input Very Low in Value column
#print(c_data[c_data['Value']>=0][c_data['Value']<=7.0]["Value"])
value_data= value_data.apply(str) 
index_list=value_data.index.copy()
for x in index_list:
    value_data[x]='very low'


value_data2=c_data[c_data['Value']>7][c_data['Value']<=14.0]["Value"]   #input  Low in Value column 
#print(c_data[c_data['Value']>=0][c_data['Value']<=7.0]["Value"])
value_data2= value_data2.apply(str) 
index_list=value_data2.index.copy()
for x in index_list:
    value_data2[x]='low'



value_data3=c_data[c_data['Value']>14][c_data['Value']<=21.0]["Value"]    #input Medium in Value column 
#print(c_data[c_data['Value']>=0][c_data['Value']<=7.0]["Value"])
value_data3= value_data3.apply(str) 
index_list=value_data3.index.copy()
for x in index_list:
    value_data3[x]='Medium'
    


value_data4=c_data[c_data['Value']>21][c_data['Value']<=28.0]["Value"]     #input High in Value column
#print(c_data[c_data['Value']>=0][c_data['Value']<=7.0]["Value"])
value_data4= value_data4.apply(str) 
index_list=value_data4.index.copy()
for x in index_list:
    value_data4[x]='High'
    
    
    

value_data5=c_data[c_data['Value']>28.0][c_data['Value']<=100]["Value"]     #input Very High in Value column
#print(c_data[c_data['Value']>=0][c_data['Value']<=7.0]["Value"])
value_data5= value_data5.apply(str) 
index_list=value_data5.index.copy()
for x in index_list:
    value_data5[x]='Very High'
    

value_data=value_data.append(value_data2) #kolonları alta alta topladık
value_data=value_data.append(value_data3)
value_data=value_data.append(value_data4)
value_data=value_data.append(value_data5)



value_data=value_data.sort_index() #indexlere göre Kolonu tekrar düzenledik

c_data["Rate Of Approval"]=value_data
c_data = c_data.drop('Value',axis=1)



c_data.Question=c_data.Question.str.replace('... if she burns the food','1') 
c_data.Question=c_data.Question.str.replace('... for at least one specific reason','2')
c_data.Question=c_data.Question.str.replace('... if she argues with him','3')
c_data.Question=c_data.Question.str.replace('... if she goes out without telling him','4')#question convert to numeric type
c_data.Question=c_data.Question.str.replace('... if she neglects the children','5')
c_data.Question=c_data.Question.str.replace('... if she refuses to have sex with him','6')


         
c_data=c_data.rename(columns={'Question': "Question Type"})       #change the convert type

#print(c_data.columns)    


marital_data=c_data[c_data['Demographics Question'].str.contains("Marital status")] # find the lines for Demographics Question
edu_data=c_data[c_data['Demographics Question'].str.contains("Education")]
emp_data=c_data[c_data['Demographics Question'].str.contains("Employment")]
age_data=c_data[c_data['Demographics Question'].str.contains("Age")]
res_data=c_data[c_data['Demographics Question'].str.contains("Residence")]



marital_data=marital_data.drop('Demographics Question', axis=1) #so if grouping  operation is success, we can get rid off unnecessarily coloumn for each dataset
marital_data=marital_data.rename(columns={'Demographics Response': "Marital Status"}) 


edu_data=edu_data.drop('Demographics Question', axis=1)
edu_data=edu_data.rename(columns={'Demographics Response': "Education"})

emp_data=emp_data.drop('Demographics Question', axis=1)
emp_data=emp_data.rename(columns={'Demographics Response': "Employment"}) 

age_data=age_data.drop('Demographics Question', axis=1)
age_data=age_data.rename(columns={'Demographics Response': "Age"})  

res_data=res_data.drop('Demographics Question', axis=1)
res_data=res_data.rename(columns={'Demographics Response': "Residence"}) 





'''
compression_opts = dict(method='zip',
                        archive_name='res_data.csv')   #create a new csv file for each dataset
res_data.to_csv('res_data.zip', index=False,
          compression=compression_opts)  



compression_opts = dict(method='zip',
                        archive_name='age_data.csv')  
age_data.to_csv('age_data.zip', index=False,
          compression=compression_opts)  


compression_opts = dict(method='zip',
                        archive_name='emp_data.csv')  
emp_data.to_csv('emp_data.zip', index=False,
          compression=compression_opts)  


compression_opts = dict(method='zip',
                        archive_name='edu_data.csv')  
edu_data.to_csv('edu_data.zip', index=False,
          compression=compression_opts)  

compression_opts = dict(method='zip',
                        archive_name='marital_data.csv')  
marital_data.to_csv('marital_data.zip', index=False,
          compression=compression_opts)  
'''






t_data=pd.read_csv("turkey.csv")

tc_data=t_data.copy()

#print(tc_data.columns)

tc_data = tc_data.drop('killingway2',axis=1)
tc_data = tc_data.drop('killingway3',axis=1)
tc_data = tc_data.drop('why2',axis=1)
tc_data = tc_data.drop('killer2',axis=1)
tc_data = tc_data.drop('date',axis=1)

tc_data=tc_data.dropna(subset=["city","age"],how="all")

#print(tc_data['city'].value_counts(dropna=False))
#print(tc_data.isnull().sum()) 
tc_data=tc_data.dropna(subset=["age"])
#print(tc_data.isnull().sum()) 

tc_data['statusofkiller'].fillna(value='Bilinmiyor',inplace=True)
tc_data['protectionorder'].fillna(value='Bilinmiyor',inplace=True)
tc_data['why1'].fillna(value='Bilinmiyor',inplace=True)
tc_data['killer1'].fillna(value='Bilinmiyor',inplace=True)
tc_data['city'].fillna(value='Bilinmiyor',inplace=True)
#print(tc_data.isnull().sum()) 

ID=np.arange(1,1321)
tc_data['ID']=ID

tc_data=tc_data.drop('id',axis=1)
tc_data=tc_data.reset_index()
tc_data=tc_data.drop('index',axis=1)


clist = list(tc_data.columns) # Get the DataFrame column names as a list
clist_new = clist[-1:]+clist[:-1]   # brings the last column in the first place
tc_data = tc_data[clist_new]

print(tc_data.columns)

tc_data.rename(columns={"city": "Sehir", 
                        "age":"Yas Durumu",
                        "protectionorder":"Koruma Karari",
                        "why1":"Oldurme Nedeni",
                        "killer1":"Katil",
                        "killingway1":"Olum Sekli",
                        "statusofkiller":"Katilin Durumu",
                        "year":"Yil"},inplace="True")
print(tc_data.columns)


'''
compression_opts = dict(method='zip',
                        archive_name='turkey_new.csv')  
tc_data.to_csv('turkey_new.zip', index=False,
          compression=compression_opts) 
'''