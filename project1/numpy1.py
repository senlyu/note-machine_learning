import numpy as np
num=[1,2,4,5]
np.mean(num)
np.median(num)
np.std(num)
num2=[1,2,3,4]
np.dot(num,num2)

import pandas as pd
from pandas import DataFrame
d={'name':pd.Series(['Braund','Cummings','Heikkinen','Allen'],index=['a','b','c','d']),
'age':pd.Series([22,38,26,35],index=['a','b','c','d']),
'fare':pd.Series([7.25,71.83,8.05],index=['a','b','c'])}
df=DataFrame(d)

print df.dtypes
print df.describe()
print df.head()
print df.tail()

# get df['name']
df['name']
df.name
# get df cols
df[['name','age','fare']]
# get df row[0]
df.iloc[[0]]
df.loc[['a']]
# get df row[3:5], 3 included 5 not
df[3:5]
# get df condition
df[(df.age>10)&(df.age==10)]

df['age'].apply(np.mean)
print df.applymap(lambda x: x>=1)

np.mean(df[df['age']>0]['fare'])


