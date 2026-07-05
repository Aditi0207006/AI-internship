x = [1,2,3,4,5]
y = [7,14,15,18,19]
from sklearn.linear_model import LinearRegression
model = LinearRegression()
import numpy as np
x= np.array(x)
y = np.array(y)
x = x.reshape(-1,1)
#train model using training
model.fit(x,y)
predict =  model.predict([[6]])
print(predict)
