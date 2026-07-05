x = [1,2,3,4,5]
y = [7,14,15,18,19]
from sklearn.linear_model import LinearRegression
model = LinearRegression()
import numpy as np
x= np.array(x).reshape(-1,1)
y = np.array(y)
model.fit(x,y)
predict =  model.predict([[6]])
print(predict)