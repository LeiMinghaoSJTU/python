import numpy as np
from math import floor
# 定义分段函数 f(x)
def f(x):
    if x == 0:
        return 1.0
    elif x == 1:
        return 1.5
    elif x == 2:
        return 2.0
    elif x == 3:
        return 2.0
    elif x == 4:
        return 2.5
    elif x == 5:
        return 2.5
    else:
        return floor(2.8*np.log2(x + 2) -2.2)/2
print(f"f0: {f(0)}") 
print(f"f1: {f(1)}") 
print(f"f2: {f(2)}") 
print(f"f3: {f(3)}")    
print(f"f4: {f(4)}")    
print(f"f5: {f(5)}")    
print(f"f6: {f(6)}") # 应约等于3.0
print(f"f7: {f(7)}") # 应约等于3.0
print(f"f8: {f(8)}") # 应约等于3.0
print(f"f9: {f(9)}") # 应约等于4.0
print(f"f10: {f(10)}")  # 应接近4.0
print(f"f11: {f(11)}")  # 应接近4.0
print(f"f12: {f(12)}")  # 应接近4.0
print(f"f13: {f(13)}")  # 应接近4.0
print(f"f14: {f(14)}")  # 应接近4.5
print(f"f15: {f(15)}")  # 应接近4.5
print(f"f16: {f(16)}")  # 应接近4.5
print(f"f17: {f(17)}")  # 应接近4.5
print(f"f18: {f(18)}")  # 应接近4.5
print(f"f19: {f(19)}")  # 应接近4.5
print(f"f20: {f(20)}")  # 应接近
