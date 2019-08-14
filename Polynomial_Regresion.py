import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset=pd.read_csv('E:\\data science\\subject vedios\\Polynomial-Linear-Regression-master\\Polynomial-Linear-Regression-master\\Position_Salaries.csv')
dataset

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values


#linear regression cassifier
from sklearn.linear_model import LinearRegression
linear_reg1=LinearRegression()
linear_reg1.fit(x,y)


#polynomial linear regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=5)
x_poly=poly_reg.fit_transform(x)

linearreg_2=LinearRegression()
linearreg_2.fit(x_poly,y)

#ploting linear regression

plt.scatter(x,y,color='red')
plt.plot(x,linear_reg1.predict(x),color='blue')
plt.show()

#ploting polynomial regression

plt.scatter(x,y,color='red')
plt.plot(x,linearreg_2.predict(x_poly),color='blue')
plt.show()
