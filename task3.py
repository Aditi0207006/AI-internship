import pandas as pd
data = pd.read_csv('bp.csv')
data.head()
y = data['Systolic BP (mm Hg)']
x = data.drop(['Systolic BP (mm Hg)','Subject_ID'], axis=1)
print(x.shape, y.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
age = int(input("Enter the age of the patient: "))
weight = int(input("Enter the weight of the patient: "))
bp = model.predict([[age, weight]])
print("Predicted systolic blood pressure:", bp[0])
from sklearn.metrics import mean_squared_error
error = mean_squared_error(y_test, model.predict(x_test))
print("Mean Squared Error:", error)
