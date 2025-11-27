import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Bengaluru_House_Data.csv")
df.head()

df = df.drop(['area_type', 'society', 'balcony', 'availability'], axis=1)
df.head()

def convert_bhk(text):
    try:
        return int(text.split()[0])
    except:
        return None

df['bhk'] = df['size'].apply(convert_bhk)
df = df.drop('size', axis=1)
df.head()


def convert_sqft(x):
    try:
        return float(x)
    except:
        if '-' in x:
            a,b = x.split('-')
            return (float(a) + float(b)) / 2
        else:
            return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
df = df.dropna()
df.head()

df['price_per_sqft'] = df['price']*100000 / df['total_sqft']
df.head()

df['location'] = df['location'].apply(lambda x: x.strip())
location_count = df['location'].value_counts()

# Any location with less than 10 entries → “other”
location_less_10 = location_count[location_count <= 10]
df['location'] = df['location'].apply(lambda x: 'other' if x in location_less_10 else x)

df.head()

df = df[(df.total_sqft / df.bhk) >= 300]
df.shape

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df = remove_pps_outliers(df)
df.shape


df = df.drop(['price_per_sqft'], axis=1)


dummies = pd.get_dummies(df.location)
df = pd.concat([df, dummies], axis=1)
df = df.drop('location', axis=1)
df.head()

X = df.drop('price', axis=1)
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)


rf = RandomForestRegressor()
rf.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("Linear Regression RMSE:", mean_squared_error(y_test, lr_pred))
print("Linear Regression R2 Score:", r2_score(y_test, lr_pred))


rf_pred = rf.predict(X_test)

print("Random Forest RMSE:", mean_squared_error(y_test, rf_pred))
print("Random Forest R2 Score:", r2_score(y_test, rf_pred))

import joblib
joblib.dump(rf, "bengaluru_price_model.pkl")

