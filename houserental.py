## House Rental Prediction ##
import pandas as pd
data = pd.read_csv('House_Rent_Dataset.csv')
data.head()
y = data['Rent']
x = data.drop(['Rent','Posted On','Floor','Area Locality'], axis=1)
import category_encoders as ce
encoder = ce.LeaveOneOutEncoder(return_df=True)
encoded_features = encoder.fit_transform(x, y)
print(encoded_features)
#min-max scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(encoded_features)
scaled_features_df = pd.DataFrame(scaled_features, columns=encoded_features.columns)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x_train, x_test, y_train, y_test = train_test_split(scaled_features_df, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
r=model.predict(x_train[1:2])
print(r)
import joblib
artifacts = {
    'encoder': encoder,
    'scaler': scaler,
    'model': model
}
joblib.dump(artifacts, 'House_rent_prediction.pkl')
encoder = artifacts['encoder']
scaler = artifacts['scaler']
model = artifacts['model']

new_house = pd.DataFrame({
    'BHK': [2],
    'Size': [1200],
    'Area Type': ['Super Area'],
    'City': ['Delhi'],
    'Furnishing Status': ['Semi-Furnished'],
    'Tenant Preferred': ['Family'],
    'Bathroom': [2],
    'Point of Contact': ['Contact Owner']
})
new_encoded = encoder.transform(new_house)
new_scaled = scaler.transform(new_encoded)
new_scaled = pd.DataFrame(
    new_scaled,
    columns=scaled_features_df.columns
)
prediction = model.predict(new_scaled)
print("Predict Rent:", prediction[0])
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import category_encoders as ce
import joblib

