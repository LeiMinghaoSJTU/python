在Python中，字典（Dictionary）是一种**可变的、无序的键值对（key-value）集合**，用于存储具有映射关系的数据。它是Python中最灵活、最常用的数据结构之一，类似于现实生活中的"字典"——通过"关键词"（键）快速查找对应的"解释"（值）。


### 一、字典的核心特点
1. **键值对结构**：每个元素由 `键（key）: 值（value）` 组成，键和值之间用冒号 `:` 分隔，不同键值对之间用逗号 `,` 分隔。
2. **键的唯一性**：字典中键必须唯一，若出现重复键，后定义的键值对会覆盖前面的。
3. **键的不可变性**：键必须是**不可变类型**（如字符串、数字、元组），列表等可变类型不能作为键。
4. **值的任意性**：值可以是任意类型（字符串、数字、列表、字典等），甚至可以是函数。
5. **无序性**：在Python 3.7之前，字典是无序的；Python 3.7+ 开始，字典会保留插入顺序（但通常仍视为无序结构，不依赖顺序进行操作）。


### 二、字典的创建
字典的创建有两种常用方式：


#### 1. 使用花括号 `{}` 直接定义
```python
# 简单字典：键为字符串，值为不同类型
person = {
    "name": "Alice",
    "age": 25,
    "is_student": False,
    "hobbies": ["reading", "hiking"]  # 值可以是列表
}

# 键为数字的字典
scores = {1: 90, 2: 85, 3: 95}
```


#### 2. 使用 `dict()` 构造函数
```python
# 方式1：通过键值对参数创建
person = dict(name="Bob", age=30, is_student=True)

# 方式2：通过包含键值对的可迭代对象（如列表、元组）创建
data = [("name", "Charlie"), ("age", 28)]
person = dict(data)  # 结果：{"name": "Charlie", "age": 28}
```


### 三、字典的基本操作


#### 1. 访问字典的值
通过**键**访问对应的值，语法：`字典名[键]`  
若键不存在，会直接报错；推荐使用 `get()` 方法，键不存在时返回默认值（默认 `None`）。

```python
person = {"name": "Alice", "age": 25}

# 直接访问（键存在时）
print(person["name"])  # 输出：Alice

# 使用get()方法（更安全）
print(person.get("age"))  # 输出：25
print(person.get("gender"))  # 键不存在，返回None
print(person.get("gender", "unknown"))  # 自定义默认值，输出：unknown
```


#### 2. 修改字典
- **添加新键值对**：直接对新键赋值。
- **修改已有键的值**：对已有键重新赋值。

```python
person = {"name": "Alice", "age": 25}

# 添加新键值对
person["gender"] = "female"
print(person)  # 输出：{"name": "Alice", "age": 25, "gender": "female"}

# 修改已有键的值
person["age"] = 26
print(person["age"])  # 输出：26
```


#### 3. 删除字典元素
- `del 字典名[键]`：删除指定键值对（键不存在会报错）。
- `字典名.pop(键)`：删除指定键值对，并返回对应的值（键不存在可指定默认值）。
- `字典名.popitem()`：删除最后插入的键值对（返回被删除的键值对元组）。
- `字典名.clear()`：清空字典中所有元素。

```python
person = {"name": "Alice", "age": 25, "gender": "female"}

# del删除
del person["gender"]
print(person)  # 输出：{"name": "Alice", "age": 25}

# pop()删除（返回值）
age = person.pop("age")
print(age)  # 输出：25
print(person)  # 输出：{"name": "Alice"}

# popitem()删除最后一个元素
person = {"name": "Alice", "age": 25}
last_item = person.popitem()
print(last_item)  # 输出：("age", 25)
print(person)  # 输出：{"name": "Alice"}

# clear()清空
person.clear()
print(person)  # 输出：{}
```


### 四、字典的常用方法
| 方法               | 说明                                         |
| ------------------ | -------------------------------------------- |
| `keys()`           | 返回所有键的视图（可迭代）                   |
| `values()`         | 返回所有值的视图（可迭代）                   |
| `items()`          | 返回所有键值对的视图（元组形式，可迭代）     |
| `update(其他字典)` | 用其他字典的键值对更新当前字典（覆盖重复键） |
| `copy()`           | 复制字典（浅拷贝）                           |


#### 示例：
```python
person = {"name": "Alice", "age": 25, "gender": "female"}

# 获取所有键
print(list(person.keys()))  # 输出：["name", "age", "gender"]

# 获取所有值
print(list(person.values()))  # 输出：["Alice", 25, "female"]

# 获取所有键值对
print(list(person.items()))  # 输出：[("name", "Alice"), ("age", 25), ("gender", "female")]

# 用新字典更新
person.update({"age": 26, "city": "Beijing"})
print(person)  # 输出：{"name": "Alice", "age": 26, "gender": "female", "city": "Beijing"}

# 复制字典
person_copy = person.copy()
print(person_copy)  # 输出：与person相同的内容
```


### 五、字典的遍历
通过 `for` 循环可遍历字典的键、值或键值对：

```python
person = {"name": "Alice", "age": 25, "gender": "female"}

# 遍历键（默认遍历键）
for key in person:
    print(key)  # 输出：name、age、gender

# 遍历值
for value in person.values():
    print(value)  # 输出：Alice、25、female

# 遍历键值对
for key, value in person.items():
    print(f"{key}: {value}")  # 输出：name: Alice、age: 25、gender: female
```


### 六、总结
字典是Python中用于存储映射关系数据的核心结构，其**键值对**设计使其能通过键快速访问、修改数据，适合存储具有属性描述的数据（如用户信息、配置参数等）。掌握字典的创建、访问、修改和遍历是Python编程的基础技能。