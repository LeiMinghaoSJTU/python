NumPy（Numerical Python）是Python中用于科学计算的核心库，其核心优势在于高效的多维数组操作和丰富的数学函数。以下是NumPy中常见的数据结构和函数介绍：


### 一、核心数据结构：`ndarray`（N-dimensional Array）
`ndarray`是NumPy的基础数据结构，即**多维数组**，具有以下特点：
- **同质性**：数组中所有元素必须是相同数据类型（如整数、浮点数）；
- **固定大小**：创建后大小不可变（与Python列表不同）；
- **高效性**：底层基于C语言实现，运算速度远快于Python原生列表。


#### 1. `ndarray`的基本属性
创建数组后，可通过以下属性获取其特征：
```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])  # 2维数组（矩阵）

print(arr.ndim)   # 维度：2
print(arr.shape)  # 形状（行数, 列数）：(2, 3)
print(arr.dtype)  # 元素类型：int64（默认）
print(arr.size)   # 总元素数：6
print(arr.itemsize)  # 每个元素的字节大小：8（int64占8字节）
```


#### 2. 数组创建函数
常用创建`ndarray`的函数：
- `np.array()`：从Python列表/元组创建数组
  ```python
  arr1 = np.array([1, 2, 3])  # 1维数组
  arr2 = np.array([[1, 2], [3, 4]], dtype=float)  # 指定类型为浮点数
  ```

- `np.zeros()`/`np.ones()`：创建全0/全1数组
  ```python
  zeros_arr = np.zeros((2, 3))  # 2行3列的全0数组
  ones_arr = np.ones((3,))     # 1维全1数组（长度3）
  ```

- `np.arange()`：类似Python的`range`，生成有序数组
  ```python
  arr = np.arange(0, 10, 2)  # 从0到10（不包含），步长2：[0, 2, 4, 6, 8]
  ```

- `np.linspace()`：生成指定区间内的等间隔数值
  ```python
  arr = np.linspace(0, 1, 5)  # 0到1之间取5个等间隔值：[0, 0.25, 0.5, 0.75, 1]
  ```

- `np.random`系列：生成随机数组
  ```python
  rand_arr = np.random.rand(2, 2)  # 2x2的[0,1)随机浮点数
  randint_arr = np.random.randint(0, 10, size=(3,))  # 3个0-10的随机整数
  ```


### 二、常用函数分类


#### 1. 数组形状操作
- `reshape()`：改变数组形状（元素总数不变）
  ```python
  arr = np.arange(6).reshape(2, 3)  # 从[0,1,2,3,4,5]变为2x3矩阵
  ```

- `flatten()`/`ravel()`：将多维数组展平为1维
  ```python
  arr = np.array([[1,2], [3,4]])
  flat_arr = arr.flatten()  # [1, 2, 3, 4]
  ```

- `concatenate()`：拼接数组（需指定轴，0为行方向，1为列方向）
  ```python
  arr1 = np.array([[1,2], [3,4]])
  arr2 = np.array([[5,6], [7,8]])
  merged = np.concatenate((arr1, arr2), axis=0)  # 行方向拼接（2x2 → 4x2）
  ```

- `split()`：拆分数组
  ```python
  arr = np.arange(6).reshape(2, 3)
  parts = np.split(arr, 2, axis=1)  # 按列拆分为2个2x1数组
  ```


#### 2. 数学运算函数
NumPy支持**向量化运算**（无需循环，直接对数组整体操作）：
- 基础运算：`+`、`-`、`*`、`/`（对应`np.add`、`np.subtract`等）
  ```python
  arr = np.array([1, 2, 3])
  print(arr + 2)  # [3, 4, 5]（每个元素加2）
  print(arr * 2)  # [2, 4, 6]（每个元素乘2）
  ```

- 统计函数：`sum()`、`mean()`、`max()`、`min()`、`std()`（标准差）等
  ```python
  arr = np.array([[1, 2], [3, 4]])
  print(arr.sum())       # 总和：10
  print(arr.mean(axis=0))  # 按列求均值：[2, 3]
  print(arr.max(axis=1))   # 按行求最大值：[2, 4]
  ```

- 其他数学函数：`np.sqrt()`（开方）、`np.exp()`（指数）、`np.log()`（对数）等
  ```python
  arr = np.array([1, 4, 9])
  print(np.sqrt(arr))  # [1, 2, 3]
  ```


#### 3. 索引与切片
`ndarray`的索引和切片类似Python列表，但支持多维操作：
```python
arr = np.arange(12).reshape(3, 4)  # 3行4列数组

# 取第2行（索引从0开始）
print(arr[1])  # [4, 5, 6, 7]

# 取第1行第2列元素
print(arr[0, 1])  # 1

# 切片：取前2行，后2列
print(arr[:2, 2:])  # [[2, 3], [6, 7]]
```


#### 4. 广播机制（Broadcasting）
当两个数组形状不同时，NumPy会自动扩展形状以匹配运算，称为“广播”：
```python
arr1 = np.array([[1, 2], [3, 4]])  # 2x2
arr2 = np.array([10, 20])          # 1x2
print(arr1 + arr2)  # 等价于 [[1+10, 2+20], [3+10, 4+20]] → [[11, 22], [13, 24]]
```


#### 5. 逻辑与条件函数
- `np.where()`：根据条件返回元素索引或替换值
  ```python
  arr = np.array([1, 2, 3, 4])
  # 找出大于2的元素索引
  print(np.where(arr > 2))  # (array([2, 3]),)
  # 替换：大于2的元素为0，否则保留原值
  print(np.where(arr > 2, 0, arr))  # [1, 2, 0, 0]
  ```

- `np.any()`/`np.all()`：判断数组中是否存在/全部满足条件
  ```python
  arr = np.array([1, 2, 3])
  print(np.any(arr > 2))  # True（存在大于2的元素）
  print(np.all(arr > 2))  # False（并非所有元素都大于2）
  ```


### 总结
NumPy的`ndarray`是高效数值计算的基础，配合其丰富的函数（数组操作、数学运算、统计分析等），可以极大简化科学计算、数据分析等场景的代码实现。掌握这些基础内容，是进一步学习Python数据科学（如Pandas、Matplotlib）的重要前提。