## https://www.tensorflow.org/tutorials/keras/basic_regression 참조
# Auto MPG 데이터셋을 사용하여 1970년대 후반과 1980년대 초반의 자동차 연비를 예측하는 모델을 생성. 
# 이 정보에는 실린더 수, 배기량, 마력(horsepower), 공차 중량 같은 속성이 포함

# 산점도 행렬을 그리기 위해 seaborn 패키지를 설치
# !pip install seaborn

from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

# 데이터셋 다운로드
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())

# 이 데이터셋은 일부 데이터가 누락되어 있다. 
print(dataset.isna().sum())

# 누락된 행을 삭제
dataset = dataset.dropna()

# "Origin" column은 수치형이 아니고 범주형이므로 원-핫 인코딩(one-hot encoding)으로 변환
origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print(dataset.tail())

# 데이터를 훈련 세트와 테스트 세트로 분할
# 테스트 세트는 모델을 최종적으로 평가할 때 사용
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 훈련 세트에서 몇 개의 열을 선택해 산점도 행렬을 생성
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()

# 전반적인 통계도 확인
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

# 특성에서 타깃 값 또는 "레이블"을 분리
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# 데이터 정규화. 정규화된 데이터를 사용하여 모델을 훈련
# 입력 데이터를 정규화하기 위해 사용한 통계치(평균과 표준편차)는 원-핫 인코딩과 마찬가지로 모델에 주입되는 모든 데이터에 적용
# 즉 테스트 세트는 물론 모델이 실전에 투입되어 얻은 라이브 데이터도 포함됨
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# 두 개의 완전 연결(densely connected) 은닉층으로 Sequential 모델을 생성. 
# 출력 층은 하나의 연속적인 값을 반환
# 나중에 두 번째 모델을 만들기 쉽도록 build_model 함수로 모델 구성 단계를 사용
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[9]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model = build_model()
model.summary()

# 결괏값의 크기와 타입을 체크하기 위해 모델을 한번 실행
# 훈련 세트에서 10개 데이터의 예제 샘플을 하나의 배치로 만들어 model.predict 메서드를 호출  --> 이상이 없음
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)

# 이 모델을 1,000번의 에포크(epoch) 동안 훈련. 훈련 정확도와 검증 정확도는 history 객체에 기록
# 에포크가 끝날 때마다 점(.)을 출력해 훈련 진행 과정을 표시합니다
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split = 0.2, verbose=0, callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

import matplotlib.pyplot as plt

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(8,12))

  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.subplot(2,1,2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

plot_history(history)

## 지정된 에포크 횟수 동안 성능 향상이 없으면 자동으로 훈련이 멈춘다.
# patience 매개변수는 성능 향상을 체크할 에포크 횟수
model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("테스트 세트의 평균 절대 오차: {:5.2f} MPG".format(mae))

# 예측 : 테스트 세트에 있는 샘플을 사용해 MPG 값을 예측
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([0, 50], [0, 50])    # (0.0) to (50, 50)
plt.show()

# 오차의 분포
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()
