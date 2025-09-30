class Solution:
    def countBits(self, n: int) -> List[int]:
        def countOnes(x: int) -> int:
            ones = 0
            while x > 0:
                x &= (x - 1) # 清除最低位的 1, 比如说 x = 1010, x-1 = 1001, x & (x-1) = 1000
                ones += 1
            return ones
        
        bits = [countOnes(i) for i in range(n + 1)] # 这种写法相当于 for i in range(n+1): bits.append(countOnes(i))
        return bits

'''这是我的原始代码'''
# class Solution:
#     def countBits(self, n: int) -> List[int]:
#         list_answer=[0]
#         for i in range(n):
#             count=0
#             s=i+1
#             while s>0.1:
#                 if s%2 != 0:
#                     count+=1
#                     s=(s-1)/2
#                 else:
#                     s=s/2
#             list_answer.append(count)
#         return list_answer