def print_separator(title):
    """打印分隔符，用于区分不同的测试部分"""
    print("\n" + "="*50)
    print(f"=== 测试: {title} ===")
    print("="*50)

def test_list_creation():
    """测试列表的创建方法"""
    print_separator("列表的创建")
    
    # 空列表
    empty_list = []
    print(f"空列表: {empty_list}, 类型: {type(empty_list)}")
    
    # 直接创建列表
    num_list = [1, 2, 3, 4, 5]
    print(f"数字列表: {num_list}")
    
    # 混合类型列表
    mixed_list = [1, "hello", True, 3.14, [6, 7]]
    print(f"混合类型列表: {mixed_list}")
    
    # 使用list()函数创建
    str_to_list = list("python")
    print(f"从字符串创建列表: {str_to_list}")
    
    tuple_to_list = list((10, 20, 30))
    print(f"从元组创建列表: {tuple_to_list}")
    
    range_to_list = list(range(5)) # range 的用法是：range(起始值, 结束值, 步长)，默认起始值为0，步长为1，结束值不包含在内
    print(f"从range创建列表: {range_to_list}")

def test_list_access():
    """测试列表元素的访问"""
    print_separator("列表元素的访问")
    
    fruits = ["apple", "banana", "cherry", "date", "elderberry"]
    print(f"测试列表: {fruits}")
    
    # 访问单个元素
    print(f"\n访问单个元素:")
    print(f"fruits[0] = {fruits[0]}")
    print(f"fruits[2] = {fruits[2]}")
    print(f"fruits[-1] = {fruits[-1]}")  # 最后一个元素
    print(f"fruits[-3] = {fruits[-3]}")  # 倒数第三个元素
    
    # 切片操作
    print(f"\n切片操作:")
    print(f"fruits[1:4] = {fruits[1:4]}")    # 索引1到3
    print(f"fruits[:3] = {fruits[:3]}")     # 开头到索引2
    print(f"fruits[2:] = {fruits[2:]}")     # 索引2到结尾
    print(f"fruits[::2] = {fruits[::2]}")   # 步长为2
    print(f"fruits[::-1] = {fruits[::-1]}") # 反转列表

def test_list_modification():
    """测试列表元素的修改"""
    print_separator("列表元素的修改")
    
    numbers = [1, 2, 3, 4, 5]
    print(f"原始列表: {numbers}")
    
    # 修改单个元素
    numbers[2] = 30
    print(f"修改索引2后的列表: {numbers}")
    
    # 切片修改
    numbers[1:4] = [20, 30, 40, 50]
    print(f"切片修改后的列表: {numbers}")
    
    # 清空列表
    numbers[:] = []
    print(f"清空后的列表: {numbers}")

def test_list_methods():
    """测试列表的常用方法"""
    print_separator("列表的常用方法")
    
    # 测试append()和extend()
    fruits = ["apple", "banana"]
    print(f"初始列表: {fruits}")
    
    fruits.append("cherry")
    print(f"append('cherry')后: {fruits}")
    
    fruits.extend(["date", "elderberry"])
    print(f"extend(['date', 'elderberry'])后: {fruits}")
    
    # 测试insert()
    fruits.insert(1, "avocado")
    print(f"insert(1, 'avocado')后: {fruits}")
    
    # 测试index()和count()
    print(f"\n'banana'的索引: {fruits.index('banana')}")
    fruits.append("apple")
    print(f"添加另一个'apple'后: {fruits}")
    print(f"'apple'出现的次数: {fruits.count('apple')}")
    
    # 测试remove()和pop()
    fruits.remove("apple")  # 移除第一个'apple'
    print(f"remove('apple')后: {fruits}")
    
    removed_item = fruits.pop(2)
    print(f"pop(2)移除的元素: {removed_item}, 列表变为: {fruits}")
    
    # 测试sort()和reverse()
    numbers = [3, 1, 4, 1, 5, 9, 2, 6]
    print(f"\n原始数字列表: {numbers}")
    
    numbers.sort()
    print(f"sort()后: {numbers}")
    
    numbers.reverse()
    print(f"reverse()后: {numbers}")
    
    # 测试copy()
    numbers_copy = numbers.copy()
    numbers_copy[0] = 0
    print(f"复制后修改的列表: {numbers_copy}")
    print(f"原始列表(未变): {numbers}")
    
    # 测试clear()
    numbers.clear()
    print(f"clear()后的列表: {numbers}")

def test_list_builtin_functions():
    """测试与列表相关的内置函数"""
    print_separator("与列表相关的内置函数")
    
    numbers = [3, 1, 4, 1, 5, 9]
    print(f"测试列表: {numbers}")
    
    print(f"len(numbers) = {len(numbers)}")
    print(f"max(numbers) = {max(numbers)}")
    print(f"min(numbers) = {min(numbers)}")
    print(f"sum(numbers) = {sum(numbers)}")
    
    # 测试in和not in
    print(f"\n3 在列表中? {3 in numbers}")
    print(f"10 在列表中? {10 in numbers}")
    print(f"10 不在列表中? {10 not in numbers}")

def test_list_iteration():
    """测试列表的遍历"""
    print_separator("列表的遍历")
    
    fruits = ["apple", "banana", "cherry", "date"]
    
    print("使用for循环遍历:")
    for fruit in fruits:
        print(f"- {fruit}")
    
    print("\n使用enumerate()获取索引和值:")
    for index, fruit in enumerate(fruits):
        print(f"- 索引 {index}: {fruit}")

def test_list_comprehension():
    """测试列表推导式"""
    print_separator("列表推导式")
    
    # 基本用法
    squares = [x**2 for x in range(10)]
    print(f"0-9的平方: {squares}")
    
    # 带条件的列表推导式
    even_numbers = [x for x in range(20) if x % 2 == 0]
    print(f"0-19的偶数: {even_numbers}")
    
    # 复杂一点的列表推导式
    words = ["apple", "banana", "cherry", "date", "elderberry"]
    long_words = [word.upper() for word in words if len(word) > 5]
    print(f"长度大于5的单词(大写): {long_words}")
    
    # 嵌套列表推导式
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    flattened = [num for row in matrix for num in row]
    print(f"二维列表展平: {flattened}")

def test_nested_lists():
    """测试嵌套列表"""
    print_separator("嵌套列表")
    
    # 创建二维列表
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    print(f"二维列表:")
    for row in matrix:
        print(row)
    
    # 访问嵌套列表元素
    print(f"\n访问元素matrix[1][2] = {matrix[1][2]}")
    
    # 修改嵌套列表元素
    matrix[0][0] = 100
    print(f"修改后matrix[0][0] = {matrix[0][0]}")
    print(f"修改后的二维列表:")
    for row in matrix:
        print(row)

def main():
    """主函数，运行所有测试"""
    print("="*70)
    print("======= Python列表常用功能测试程序 =======")
    print("="*70)
    
    test_list_creation()
    test_list_access()
    test_list_modification()
    test_list_methods()
    test_list_builtin_functions()
    test_list_iteration()
    test_list_comprehension()
    test_nested_lists()
    
    print("\n" + "="*70)
    print("======= 所有测试完成 =======")
    print("="*70)

if __name__ == "__main__":
    main()
