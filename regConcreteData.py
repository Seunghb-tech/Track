#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras.utils import to_categorical
from collections import Counter
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_excel('Concrete_data.xls')
print(df.head())
print(df.describe())

print(df.columns)
df.rename(columns={'Cement (component 1)(kg in a m^3 mixture)':'cement',
       'Blast Furnace Slag (component 2)(kg in a m^3 mixture)':'blast',
       'Fly Ash (component 3)(kg in a m^3 mixture)':'fly',
       'Water  (component 4)(kg in a m^3 mixture)':'water',
       'Superplasticizer (component 5)(kg in a m^3 mixture)':'super',
       'Coarse Aggregate  (component 6)(kg in a m^3 mixture)':'corse',
       'Fine Aggregate (component 7)(kg in a m^3 mixture)':'fine', 
       'Age (day)':'age',
        'Concrete compressive strength(MPa, megapascals) ':'strength'}, inplace=True)
print(df.head())

X = df.drop(['strength'], axis=1)
print(X.head())

Y = df['strength']
print(Y.head())

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
print(X.shape)

# sns.pairplot(df)
# plt.show()

X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.1)
print(X_train.shape)

model = Sequential()
model.add(Dense(256, input_shape=(8,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss='mse', optimizer='adam')
model.summary()

hist = model.fit(X_train, Y_train, epochs=100, validation_split=0.1, verbose=1 )

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.plot(hist.history['loss'], color='r')
plt.plot(hist.history['val_loss'], color='b')
plt.title('Loss')
plt.show()

score = model.evaluate(X_test, Y_test)
print(score)

pred = model.predict(X_test[-5:])
print(pred)
print(Y_test[-5:])
