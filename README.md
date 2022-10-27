AIM

To read the given data and perform Feature Transformation process and save the data to a file.

EXPLANATION

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

ALGORITHM

STEP 1 Read the given Data

STEP 2 Clean the Data Set using Data Cleaning Process

STEP 3 Apply Feature Transformation techniques to all the features of the data set

STEP 4 Save the data to the file CODE:

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

df=pd.read_csv("/content/data.csv")

print(df)

df.head()

df.isnull().sum()

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')

plt.show()

sm.qqplot(df.HighlyNegativSkew,fit=True,line='45')

sm.qqplot(df.ModeratPositiveSkew,fit=True,line='45')

plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')

plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew

sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()

df4=df.copy()

df4['ModerateNegativeSkew_1'],parameters=stats.yeojohnson(df4.ModerateNegativeSkew)

sm.qqplot(df4.ModerateNegativeSkew_1,fit=True,line='45')

plt.show()

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df4['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df4[['ModerateNegativeSkew']]))

sm.qqplot(df4['ModerateNegativeSkew_2'],fit=True,line='45')

plt.show()
![image](https://user-images.githubusercontent.com/115688029/198326772-17c076f7-5812-4056-a75f-04a5f1416e56.png)
![image](https://user-images.githubusercontent.com/115688029/198326983-7557debe-d4e9-4550-a460-7c8689f06e5e.png)
![image](https://user-images.githubusercontent.com/115688029/198327120-87f9c95b-12ad-4f85-aab6-e196cf8b8ec6.png)
![image](https://user-images.githubusercontent.com/115688029/198327289-e8b3d700-c1a9-49d2-89af-37a441c9cea6.png)
![image](https://user-images.githubusercontent.com/115688029/198327433-746cbd23-1851-4a4b-83df-58b9ec14aeb1.png)
![image](https://user-images.githubusercontent.com/115688029/198327536-1c3dbb46-03e1-4e6f-ac24-9b7be42e9771.png)
![image](https://user-images.githubusercontent.com/115688029/198327692-a43b0bff-2811-45d8-a288-4e2de98c0034.png)
![image](https://user-images.githubusercontent.com/115688029/198327907-7d77c7c4-3806-41e8-8760-eb35f86417cc.png)
![image](https://user-images.githubusercontent.com/115688029/198328080-f073b7d9-5168-4fc9-8b7e-ec7ce1e94ff3.png)
![image](https://user-images.githubusercontent.com/115688029/198328333-b68d3f24-db85-4e76-96ec-12b85dcaf0f0.png)
![image](https://user-images.githubusercontent.com/115688029/198328614-c20ccacc-24f7-4480-8706-3c18ad814783.png)
![image](https://user-images.githubusercontent.com/115688029/198328862-f0227888-c54a-49a7-94bb-dbe66f021d30.png)
![image](https://user-images.githubusercontent.com/115688029/198329011-b5c008f8-f48d-4089-a366-9e4923e2bc2b.png)










