import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Reshape, Lambda, Bidirectional, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import cv2
import os
import numpy as np

# ディレクトリにある画像ファイルを読み込む関数
def load_images_from_directory(directory, img_height, img_width):
    images = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # グレースケールで読み込む
        img = cv2.resize(img, (img_width, img_height))    # 必要であればサイズを変更
        img = img.astype('float32') / 255.0               # ピクセル値を0〜1に正規化
        images.append(img)
    images = np.expand_dims(np.array(images), axis=-1)    # チャンネルを追加して形を合わせる
    return images

train_images = load_images_from_directory('path/to/train/images', height, width)

# 入力層の定義
input_img = Input(shape=(height, width, 1), name='image_input')

# CNNブロック
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# 特徴量を1次元に変換
x = Reshape(target_shape=(new_height, new_width))(x)

# LSTMブロック (双方向)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)

# 出力層
output = Dense(num_classes, activation='softmax')(x)

# CTCロスを定義するLambdaレイヤー
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

labels = Input(name='label_input', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([output, labels, input_length, label_length])

# モデル定義
model = Model(inputs=[input_img, labels, input_length, label_length], outputs=ctc_loss)

# コンパイル
model.compile(optimizer=Adam())

# モデルの訓練
model.fit([train_images, train_labels, train_input_length, train_label_length], 
          train_labels, epochs=10, batch_size=32)
