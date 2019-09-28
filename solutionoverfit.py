# https://www.tensorflow.org/tutorials/keras/overfit_and_underfit 참조
# 과대적합(overfitting)과 과소적합(underfitting) 사이에서 균형을 잡아야 한다.
# 과대적합을 막는 가장 좋은 방법은 더 많은 훈련 데이터를 사용하는 것
# 데이터를 더 준비할 수 없을 때 그다음으로 가장 좋은 방법은 규제(regularization)와 같은 기법을 사용하는 것
# 널리 사용되는 두 가지 규제 기법인 가중치 규제와 드롭아웃(dropout)
# 이런 기법을 사용하여 IMDB 영화 리뷰 분류 모델의 성능을 향상시켜 보자.

from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# IMDB 데이터셋 다운로드
# 여기서는 문장을 멀티-핫 인코딩(multi-hot encoding)으로 변환
# 시퀀스 [3, 5]를 인덱스 3과 5만 1이고 나머지는 모두 0인 10,000 차원 벡터로 변환
NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
    # 0으로 채워진 (len(sequences), dimension) 크기의 행렬을 만듭니다
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # results[i]의 특정 인덱스만 1로 설정합니다
    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

# 단어 인덱스는 빈도 순으로 정렬되어 있습니다. 그래프에서 볼 수 있듯이 인덱스 0에 가까울수록 1이 많이 등장
plt.plot(train_data[0])
plt.show()

# 과대적합을 막는 가장 간단한 방법은 모델의 규모를 축소하는 것. 즉, 모델에 있는 학습 가능한 파라미터의 수를 줄인다.
# 알맞은 모델의 크기를 찾으려면 비교적 적은 수의 층과 파라미터로 시작해서 검증 손실이 감소할 때까지 새로운 층을 추가하거나 층의 크기를 늘리는 것이 좋다. 
# 영화 리뷰 분류 네트워크를 사용해 이를 실험해 보자.

# Dense 층만 사용하는 간단한 기준 모델을 만들고 작은 규모의 버전와 큰 버전의 모델을 만들어 비교

# 기준 모델 만들기
baseline_model = keras.Sequential([
    # `.summary` 메서드 때문에 `input_shape`가 필요
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

baseline_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

baseline_model.summary()

baseline_history = baseline_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)

# 작은 모델 만들기
smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

smaller_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

smaller_model.summary()

smaller_history = smaller_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)

# 큰 모델 만들기
bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

bigger_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','binary_crossentropy'])

bigger_model.summary()

bigger_history = bigger_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)

# 훈련 손실과 검증 손실 그래프 그리기, 실선은 훈련 손실이고 점선은 검증 손실
def plot_history(histories, key='binary_crossentropy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key], '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])
  plt.show()


plot_history([('baseline', baseline_history), ('smaller', smaller_history), ('bigger', bigger_history)])

# 작은 네트워크가 기준 모델보다 더 늦게 과대적합이 시작되었다. 
# 큰 네트워크는 거의 바로 첫 번째 에포크 이후에 과대적합이 시작되고 훨씬 더 심각하게 과대적합된다. 
# 네트워크의 용량이 많을수록 훈련 세트를 더 빠르게 모델링하고 훈련 손실이 낮아진다. 하지만 더 쉽게 과대적합된다(훈련 손실과 검증 손실 사이에 큰 차이가 발생)

# 전략 1 --> 가중치를 규제하기
# 간단한 모델은 복잡한 것보다 과대적합되는 경향이 작다.

# 따라서 가중치가 작은 값을 가지도록 네트워크의 복잡도에 제약을 가하는 것. 이는 가중치 값의 분포를 좀 더 균일하게 만들어 준다. 
# 이를 "가중치 규제"(weight regularization). 네트워크의 손실 함수에 큰 가중치에 해당하는 비용을 추가. 이 비용은 두 가지 형태가 있다:
## L1 규제는 가중치의 절댓값에 비례하는 비용을 추가(즉, 가중치의 "L1 노름(norm)"을 추가)
## L2 규제는 가중치의 제곱에 비례하는 비용을 추가(즉, 가중치의 "L2 노름"의 제곱을 추가). 신경망에서는 L2 규제를 가중치 감쇠(weight decay)

# tf.keras에서는 가중치 규제 객체를 층의 키워드 매개변수에 전달하여 가중치에 규제를 추가한다. 
# l2(0.001)는 가중치 행렬의 모든 값이 0.001 * weight_coefficient_value**2만큼 더해진다는 의미. 이런 페널티(penalty)는 훈련할 때만 추가된다.

l2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

l2_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

l2_model_history = l2_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)

# L2 규제의 효과를 확인해 보자.
plot_history([('baseline', baseline_history), ('l2', l2_model_history)])

# 결과에서 보듯이 모델 파라미터의 개수는 같지만 L2 규제를 적용한 모델이 기본 모델보다 과대적합에 훨씬 잘 견디고 있다.

# 전략 2 --> 드롭아웃 추가하기
# 드롭아웃(dropout)은 신경망에서 가장 효과적이고 널리 사용하는 규제 기법 중 하나. "드롭아웃 비율"은 0이 되는 특성의 비율. 보통 0.2~0.5
# 테스트 단계에서는 어떤 유닛도 드롭아웃하지 않는다. 훈련 단계보다 더 많은 유닛이 활성화되기 때문에 균형을 맞추기 위해 층의 출력 값을 드롭아웃 비율만큼 줄인다.

dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

dpt_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','binary_crossentropy'])

dpt_model_history = dpt_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)

plot_history([('baseline', baseline_history),('dropout', dpt_model_history)])

# 드롭아웃을 추가하니 기준 모델보다 확실히 향상되었다.

# 정리하면 신경망에서 과대적합을 방지하기 위해 가장 널리 사용하는 방법은 다음과 같다:
## 더 많은 훈련 데이터를 모은다.
## 네트워크의 용량을 줄인다.
## 가중치 규제를 추가.
## 드롭아웃을 추가.
## 다른 중요한 방법 두 가지는 데이터 증식(data-augmentation)과 배치 정규화(batch normalization)