import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle

df=pd.read_csv('salaries.csv')
df['experience'].fillna(0,inplace=True)
df['test_score'].fillna(df['test_score'].mean(),inplace=True)

x=df.iloc[:,:-1]
y=df.iloc[:,-1]

def convert_to_int(word):
    word_dict={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'zero':0,0:0}
    return word_dict[word]
x['experience']=x['experience'].apply(lambda x:convert_to_int(x))

lr=LinearRegression()
lr.fit(x,y)

#saving model to disk
pickle.dump(lr,open('model.pkl','wb'))
#loading model to compare 
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))
