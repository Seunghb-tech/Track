# 이 튜토리얼에서는 운동화나 셔츠 같은 옷 이미지를 분류하는 신경망 모델을 훈련
# 10개의 범주(category)와 70,000개의 흑백 이미지로 구성된 패션 MNIST 데이터셋을 사용하겠습니다. 이미지는 해상도(28x28 픽셀)

from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

# tensorflow와 tf.keras를 임포트합니다
import tensorflow as tf
from tensorflow import keras

# 헬퍼(helper) 라이브러리를 임포트합니다
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# 네트워크를 훈련하는데 60,000개의 이미지를 사용. 10,000개의 이미지로 평가
# load_data() 함수를 호출하면 4 개의 넘파이(NumPy) 배열이 반환됨
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 이미지는 28x28 크기의 넘파이 배열이고 픽셀 값은 0과 255 사이. 레이블(label)은 0에서 9까지의 정수 배열로 옷의 클래스(class)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 데이터 탐색

print(train_images.shape)

print(len(train_labels))

print(train_labels)

print(test_images.shape)

print(len(test_labels))

# 데이터 전처리. 픽셀 값의 범위가 0~255
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# 이 값의 범위를 0~1 사이로 조정
train_images = train_images / 255.0

test_images = test_images / 255.0

# 훈련 세트에서 처음 25개 이미지와 그 아래 클래스 이름을 출력해 보자.
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# 모델 구성
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
# 훈련 세트에서 약 0.88(88%) 정도의 정확도를 달성한다.
model.fit(train_images, train_labels, epochs=5)

# 정확도 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('테스트 정확도:', test_acc)

# 테스트 세트의 정확도가 훈련 세트의 정확도보다 조금 낮다. 훈련 세트의 정확도와 테스트 세트의 정확도 사이의 차이는 과대적합(overfitting) 때문임

# 예측 만들기 --> 이 예측은 10개의 숫자 배열. 이 값은 10개의 옷 품목에 상응하는 모델의 신뢰도(confidence)
predictions = model.predict(test_images)
print(predictions[0])

# 가장 높은 신뢰도를 가진 레이블을 찾기
print(np.argmax(predictions[0]))

# 모델은 이 이미지가 앵클 부츠(class_name[9])라고 가장 확신
# 이 값이 맞는지 테스트 레이블을 확인해 보자.
print(test_labels[0])

# 10개의 신뢰도를 모두 그래프로 표현
# 올바르게 예측된 레이블은 파란색이고 잘못 예측된 레이블은 빨강색
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# 처음 X 개의 테스트 이미지와 예측 레이블, 진짜 레이블을 출력
# 올바른 예측은 파랑색으로 잘못된 예측은 빨강색
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# 마지막으로 훈련된 모델을 사용하여 한 이미지에 대한 예측을 만든다.
# 테스트 세트에서 이미지 하나를 선택
img = test_images[0]
print(img.shape)

# tf.keras 모델은 한 번에 샘플의 묶음 또는 배치(batch)로 예측을 만드는데 최적화되어 있다. 하나의 이미지를 사용할 때에도 2차원 배열로 만들어야 한다.
# 이미지 하나만 사용할 때도 배치에 추가
img = (np.expand_dims(img,0))

print(img.shape)

# 이미지의 예측을 실시
predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()

# model.predict는 2차원 넘파이 배열을 반환하므로 첫 번째 이미지의 예측을 선택
prediction_result = np.argmax(predictions_single[0])
print(prediction_result)

