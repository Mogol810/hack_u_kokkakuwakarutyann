from keras import layers
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.datasets import cifar10
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import os
import pickle


#ハイパーパラメータ
batch_size = 64
epochs = 8


#入力画像のパラメータ
img_width = 320
img_height = 240
#データ格納先のパス
save_data_dir_path = "C:\\Users\\cryto\\Desktop\\hack u\\ex_data"
#なかった時にファイルを作る
os.makedirs(save_data_dir_path, exist_ok=True)

# グラフ画像のサイズ
FIG_SIZE_WIDTH = 12
FIG_SIZE_HEIGHT = 10
FIG_FONT_SIZE = 25

data_x = []
data_y = []
num_classes = 3

#教師データ
#ナチュラルを0
#ストレートを1
#ウェーブを2

#ナチュラルの画像群をロード
filepath = os.getcwd()+"\\dataset\\train\\train_n"
for i in range(597):
    filename = "\\train_n"+str(i)+".jpg"
    img = img_to_array(load_img(filepath+filename, grayscale=True, target_size=(img_width,img_height)))
    data_x.append(img)
    data_y.append(0)

#ストレートの画像群をロード
filepath = os.getcwd()+"\\dataset\\train\\train_s"
for i in range(857):
    filename = "\\train_s"+str(i)+".jpg"
    img = img_to_array(load_img(filepath+filename, grayscale=True, target_size=(img_width,img_height)))
    data_x.append(img)
    data_y.append(1)

#ウェーブの画像群をロード
filepath = os.getcwd()+"\\dataset\\train\\train_w"
for i in range(567):
    filename = "\\train_w"+str(i)+".jpg"
    img = img_to_array(load_img(filepath+filename, grayscale=True, target_size=(img_width,img_height)))
    data_x.append(img)
    data_y.append(2)

test_data_x =[]
test_data_y =[]

#ナチュラルのテスト用データをロード
filepath = os.getcwd()+"\\dataset\\test\\test_n"
for i in range(265):
    filename = "\\test_n"+str(i)+".jpg"
    img = img_to_array(load_img(filepath+filename, grayscale=True, target_size=(img_width,img_height)))
    test_data_x.append(img)
    test_data_y.append(0)
#ストレートのテスト用データをロード
filepath = os.getcwd()+"\\dataset\\test\\test_s"
for i in range(341):
    filename = "\\test_s"+str(i)+".jpg"
    img = img_to_array(load_img(filepath+filename, grayscale=True, target_size=(img_width,img_height)))
    test_data_x.append(img)
    test_data_y.append(0)

#ウェーブのテスト用データをロード
filepath = os.getcwd()+"\\dataset\\test\\test_w"
for i in range(201):
    filename = "\\test_w"+str(i)+".jpg"
    img = img_to_array(load_img(filepath+filename, grayscale=True, target_size=(img_width,img_height)))
    test_data_x.append(img)
    test_data_y.append(0)

#numpy配列に変換
x_train = np.array(data_x)
y_train = np.array(data_y)
x_test = np.array(test_data_x)
y_test = np.array(test_data_y)

#正規化
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

#正解ラベルをone hotデータにエンコーディング
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(x_train.shape, 'x train samples')
print(x_test.shape, 'x test samples')
print(y_train.shape, 'y train samples')
print(y_test.shape, 'y test samples')

#モデルの構築
model = Sequential()
#入力層
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(img_width,img_height,1)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(9,9),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(Dropout(.2))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(Dropout(.3))
model.add(layers.Dense(3,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-3), metrics=['accuracy'])

history = model.fit(x_train, 
                        y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        verbose=1, 
                        )

score = model.evaluate(x_test, 
                            y_test,
                            verbose=0
                            )

print('Test loss:', score[0])

print('Test accuracy:', score[1])

# モデル構造の保存
open(save_data_dir_path  + "model.json","w").write(model.to_json())  

# 学習済みの重みを保存
model.save_weights(save_data_dir_path + "weight.hdf5")

# 学習履歴を保存
with open(save_data_dir_path + "history.json", 'wb') as f:
    pickle.dump(history.history, f)

