import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

df = pd.read_csv("weatherAUS.csv")
df["Date"] = pd.to_datetime(df["Date"], format = "%Y-%m-%dT", errors = "coerce")
df["Date_month"] = df["Date"].dt.month
df["Date_day"] = df["Date"].dt.day
df1 = df.drop(["Date","MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine","WindGustDir","WindGustSpeed","WindDir9am"],axis=1)
df1=df1.drop(["WindDir3pm","WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm","Pressure9am","Pressure3pm"],axis=1)
df1=df1.drop(["Cloud9am","Cloud3pm","Temp9am","Temp3pm","RainToday"],axis=1)
df=df1
dummies = pd.get_dummies(df.Location)
dummies.head(3)
df = pd.concat([df,dummies],axis='columns')
df.head()
df.drop('Location',axis=1,inplace=True)
df["RainTomorrow"] = pd.get_dummies(df["RainTomorrow"], drop_first = True)
X = df.drop(["RainTomorrow"], axis=1)
y = df["RainTomorrow"]
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
x_train = np.reshape(x_train,(101822,1,51))
x_test = np.reshape(x_test,(43638,1,51))
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1,51)),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(50,activation='relu'),
    keras.layers.Dense(2,activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10)
y_predicted = model.predict(x_test)
y_predicted_labels = [i[1] for i in y_predicted]
y_predicted_labels[:10]