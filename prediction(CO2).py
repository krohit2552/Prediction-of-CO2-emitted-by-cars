 import pandas as pd
df=pd.read_csv("D:\data1.csv")
df.head(20)

df.shape

df.describe()

X=df[['Weight','Volume']]
X.head()

y=df['CO2']
y.head()

from sklearn import linear_model
regr=linear_model.LinearRegression() 
regr.fit(X,y)


# predict the CO2 emission of a car where the weight is 2100kg, and the volume is 1100cm3:

predictedCO2=regr.predict([[2100,1100]])
print(predictedCO2)

# REMOVE WARNING

import warnings
warnings.filterwarnings('ignore')

#  coefficient values of the regression object:

print(regr.coef_)

# Explaination of coefficient values of the regression object

# array represents the coefficient values of weight and volume.

# Weight: 0.00755095
# Volume: 0.00780526

# These values tell us that if the weight increase by 1000kg, the CO2 emission increases by 0.00755095g*1000=7.55095g.

# And if the engine size (Volume) increases by 1000cm3, the CO2 emission increases by 0.00780526 g*1000=7.80526g.

# predict the CO2 emission of a car where the weight is 3100kg, and the volume is 1100cm3:

predictedCO2=regr.predict([[3100,1100]])

print(predictedCO2)

# Predict the CO2 emission of a car where the weight is 3100kg, and the volume is 2100cm3. here emission of co2 is increase by 7.80526g

predictedCO2=regr.predict([[3100,2100]])
print(predictedCO2)

