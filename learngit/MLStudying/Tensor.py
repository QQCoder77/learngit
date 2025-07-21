import tensorflow as tf
import numpy as np
#Normal way to create tensor
#From Python
tensor_from_list=tf.constant([[1,2],[3,4]])
#NumPy
numpy_array=np.array([[5,6],[7,8]])
tensor_from_numpy=tf.constant(numpy_array)
# 全零张量
zeros = tf.zeros((2, 3))  # 2行3列的全0矩阵

# 全一张量
ones = tf.ones((3, 2))    # 3行2列的全1矩阵

# 单位矩阵
eye = tf.eye(3)           # 3×3的单位矩阵

# 填充特定值
filled = tf.fill((2, 2), 7)  # 2×2矩阵，所有元素为7
print (tensor_from_list)