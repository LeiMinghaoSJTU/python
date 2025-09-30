from collections import Counter

class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool:
        # 字符集必须相同
        if set(word1) != set(word2):
            return False
        # 频率分布的多重集必须相同（排序后频率列表相等）
        return sorted(Counter(word1).values()) == sorted(Counter(word2).values())
    '''
    Counter 是 Python 标准库 collections 中的一个工具类，用于统计可迭代对象中元素的出现次数。当传入字符串 word1 时，它会返回一个类似字典的对象，其中：
    键（key） 是 word1 中的每个独特字符；
    值（value） 是该字符在 word1 中出现的次数（频率）。
    例如：若 word1 = "aabbc"，则 Counter(word1) 的结果为：Counter({'a': 2, 'b': 2, 'c': 1})
    sorted() 是 Python 内置函数，用于对可迭代对象进行升序排序，并返回一个新的列表。
    '''