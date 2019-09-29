# Pandas dataframes을 tf.data.Dataset에 로드하는 방법에 대한 예
# Cleveland Clinic Foundation for Heart Disease에서 제공하는 작은 데이터 세트을 사용
# CSV에는 수백 개의 행이 있다. 각 행은 환자를 설명하고, 각 열은 속성을 설명한다. 
# 이 정보를 사용해 환자가 심장병에 걸렸는지 여부를 예측하는 것임. 즉, binary classification 작업임

# 여기서는 2가지 방법을 사용해 모델을 구성하고 학습을 실시하였음

# Commented out IPython magic to ensure Python compatibility.
from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import tensorflow as tf

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')

df = pd.read_csv(csv_file)

print(df.head())
print(df.dtypes)

# thal 열을 이산 수치값으로 변환
df['thal'] = pd.Categorical(df['thal'])
# print(df['thal'])
df['thal'] = df.thal.cat.codes
# print(df['thal'])

print(df.head())
# print(df.dtypes)

target = df.pop('target')
print(df.head())
# print(target.head())

# tf.data.Dataset을 사용한 data 로드
# pandas dataframe으로부터 값을 읽기 위해 tf.data.Dataset.from_tensor_slices를 사용한다.
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
print(type(dataset))

# for feat, targ in dataset.take(5):
#   print('Features: {}, Target: {}'.format(feat, targ))

tf.constant(df['thal'])    # tensor("Const:0", shape=(303,), dtype=int8)

# dataset을 섞은 후 1개씩 배치
train_dataset = dataset.shuffle(len(df)).batch(1)

# 모델을 생성하고 학습하기
def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model

model = get_compiled_model()
model.fit(train_dataset, epochs=15)


# 특징(피쳐) 열에 대한 또 다른 방법

# 모델에 입력으로 사전(dict)을 전달하는 것은 함수 APIs를 사용해 전처리를 적용하고 층이 쌓을 수 있어
# tf.keras.layer.Input 층의 매칭 dict을 만드는 것이 매우 쉽다.
inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
x = tf.stack(list(inputs.values()), axis=-1)

x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model_func = tf.keras.Model(inputs=inputs, outputs=output)

model_func.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)

# for dict_slice in dict_slices.take(1):
#   print (dict_slice)

model_func.fit(dict_slices, epochs=15)

