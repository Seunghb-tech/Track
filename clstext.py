# 영화 리뷰(review) 텍스트를 긍정(positive) 또는 부정(negative)으로 분류
# Internet Movie Database에서 수집한 50,000개의 영화 리뷰 텍스트를 담은 IMDB 데이터셋을 사용
# 25,000개 리뷰는 훈련용으로, 25,000개는 테스트용으로 나뉘어져 있다. 클래스는 균형이 잡혀 있다. 즉 긍정적인 리뷰와 부정적인 리뷰의 개수가 동일

# keras.datasets.imdb is broken in 1.13 and 1.14, by np 1.16.3
# !pip install -q tf_nightly

from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals
import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

# MDB 데이터셋은 텐서플로와 함께 제공된다. 리뷰(단어의 시퀀스(sequence))는 미리 전처리해서 정수 시퀀스로 변환되어 있다. 
# 각 정수는 어휘 사전에 있는 특정 단어를 의미

# IMDB 데이터셋을 컴퓨터에 다운로드
# 매개변수 num_words=10000은 훈련 데이터에서 가장 많이 등장하는 상위 10,000개의 단어를 선택. 데이터 크기를 적당하게 유지하기 위해 드물에 등장하는 단어는 제외
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 이 데이터셋의 샘플은 전처리된 정수 배열. 이 정수는 영화 리뷰에 나오는 단어를 나타냄
# 레이블(label)은 정수 0 또는 1. 0은 부정적인 리뷰이고 1은 긍정적인 리뷰
print("훈련 샘플: {}, 레이블: {}".format(len(train_data), len(train_labels)))

# 리뷰 텍스트는 어휘 사전의 특정 단어를 나타내는 정수로 변환되어 있다. 첫 번째 리뷰로 확인
print(train_data[0])

# 영화 리뷰들은 길이가 다르다. 다음 코드는 첫 번째 리뷰와 두 번째 리뷰에서 단어의 개수를 출력
# 신경망의 입력은 길이가 같아야 하기 때문에 나중에 이 문제를 해결하도록 하자.
print(len(train_data[0]), len(train_data[1]))

# 정수를 단어로 다시 변환하기
# 정수와 문자열을 매핑한 딕셔너리(dictionary) 객체에 질의하는 헬퍼(helper) 함수를 만들자.

# 단어와 정수 인덱스를 매핑한 딕셔너리
word_index = imdb.get_word_index()

# word_index

# 처음 몇 개 인덱스는 사전에 정의되어 있다
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# word_index

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# decode_review 함수를 사용해 첫 번째 리뷰 텍스트를 출력
decode_review(train_data[0])

# 데이터 준비
# 리뷰-정수 배열-는 신경망에 주입하기 전에 텐서로 변환되어야 한다. 변환하는 방법에는 몇 가지가 있다:
## 원-핫 인코딩(one-hot encoding)은 정수 배열을 0과 1로 이루어진 벡터로 변환. 예를 들어 배열 [3, 5]을 인덱스 3과 5만 1이고 나머지는 모두 0인 10,000차원 벡터로 변환
# 이 방법은 num_words * num_reviews 크기의 행렬이 필요하기 때문에 메모리를 많이 사용

# 다른 방법으로는, 정수 배열의 길이가 모두 같도록 패딩(padding)을 추가해 max_length * num_reviews 크기의 정수 텐서를 만든다. 
# 이런 형태의 텐서를 다룰 수 있는 임베딩(embedding) 층을 신경망의 첫 번째 층으로 사용할 수 있다.

# 여기서는 두 번째 방식을 사용

# 영화 리뷰의 길이가 같아야 하므로 pad_sequences 함수를 사용해 길이를 맞춘다.
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)

# 샘플의 길이를 확인
len(train_data[0]), len(train_data[1])

# (패딩된) 첫 번째 리뷰 내용을 확인
print(train_data[0])

# 모델 구성
# 이 예제의 입력 데이터는 단어 인덱스의 배열. 예측할 레이블은 0 또는 1

# 입력 크기는 영화 리뷰 데이터셋에 적용된 어휘 사전의 크기(10,000개의 단어)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None,)))
# 첫번째 인자 = 단어 집합의 크기. 즉, 총 단어의 개수
# 두번째 인자 = 임베딩 벡터의 출력 차원. 결과로서 나오는 임베딩 벡터의 크기
# input_length = 입력 시퀀스의 길이
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

# 첫 번째 층은 Embedding 층. 이 층은 정수로 인코딩된 단어를 입력 받고 각 단어 인덱스에 해당하는 임베딩 벡터를 찾는다. 이 벡터는 모델이 훈련되면서 학습
# 이 벡터는 출력 배열에 새로운 차원으로 추가된다. 최종 차원은 (batch, sequence, embedding)
# GlobalAveragePooling1D 층은 sequence 차원에 대해 평균을 계산하여 각 샘플에 대해 고정된 길이의 출력 벡터를 반환. 길이가 다른 입력을 다루는 가장 간단한 방법
# 이 고정 길이의 출력 벡터는 16개의 은닉 유닛을 가진 완전 연결(fully-connected) 층(Dense)을 거친다.
# 마지막 층은 하나의 출력 노드(node)를 가진 완전 연결 층. sigmoid 활성화 함수를 사용하여 0과 1 사이의 실수를 출력

# 손실 함수와 옵티마이저
model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['acc'])

# 원본 훈련 데이터에서 10,000개의 샘플을 떼어내어 검증 세트(validation set)를 만들기
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 이 모델을 512개의 샘플로 이루어진 미니배치(mini-batch)에서 40번의 에포크(epoch) 동안 훈련
# 훈련하는 동안 10,000개의 검증 세트에서 모델의 손실과 정확도를 모니터링
history = model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# 모델 평가
results = model.evaluate(test_data, test_labels)

print(results)

# 정확도와 손실 그래프 그리기
history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo"는 "파란색 점"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b는 "파란 실선"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()   # 그림을 초기화

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 약 20번째 에포크 이후가 최적점인 것 같다. 이는 과대적합 때문이다.

