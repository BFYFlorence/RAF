import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
import numpy as np
import os

'''3 chapter
(x, y), (x_val, y_val) = datasets.mnist.load_data()
# print((x,y)[0][0])
# 转换为浮点张量，并缩放到 -1~1, 255.的意思是其是一位浮点数，-1是减1
x = 2*tf.convert_to_tensor(x, dtype=tf.float32)/255.-1
# 转换为整形张量
y = tf.convert_to_tensor(y, dtype=tf.int32)
print(x.shape, y.shape)

# 构建数据集对象
train_dataset = tf.data.Dataset.from_tensor_slices((x,y))
# 批量训练
train_dataset = train_dataset.batch(512)

# 设计网络模型
model = keras.Sequential([layers.Dense(256, activation='relu'),
                          layers.Dense(128, activation='relu'),
                          layers.Dense(10)])

# 构建梯度记录环境
with tf.GradientTape() as tape:
    # 打平操作, [b, 28, 28] => [b, 784]
    x = tf.reshape(x, (-1, 28*28))
    # step1. 得到模型输出output [b, 784] => [b, 10]
    out = model(x)
    # [b] => [b, 10] # one-hot编码
    y_onehot = tf.one_hot(y, depth=10)
    # 计算差的平方和, [b, 10]
    loss = tf.square(out-y_onehot)
    # 计算每个样本的平均误差
    loss = tf.reduce_sum(loss) / x.shape[0]
    # 自动计算梯度
    grads = tape.gradient(loss, model.trainable_variables)
    # 更新网络参数
    optimizer = tf.keras.optimizers.RMSprop(0.001)  # 创建优化器，指定学习率
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
'''
'''
# 创建标量
a = tf.constant(1.2)
print(type(a), tf.is_tensor(a))
print(a)

# 创建向量
a = tf.constant([1.2,2.0,3.1])
print(a,a.shape)

# 定义矩阵
a = tf.constant([[1,2],[3,4]])
print(a,a.shape)

# 定义字符串
a = tf.constant("Hello, Deep Learning.")
print(a,a.shape)

x = tf.random.normal([2,4]) # 2个样本，特征长度为4的张量
w = tf.ones([4,3]) # 定义W张量
b = tf.zeros([3]) # 定义b张量
o = x@w+b # X@W+b运算'''
x = tf.random.normal([4,32,32,3])
print(x[0,:])

