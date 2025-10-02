'''我的解法'''
from collections import deque
class Solution:
    def numTilings(self, n: int) -> int:
        if n <= 2:
            return n
        elif n == 3:
            return 5
        else:
            sum_queue = deque([1,3,8]) # 储存 S_{n-3} 到 S_{n-1}，其中 S_n = f(n) + S_{n-1}, f(n)为完全覆盖的方案数
            f = 0
            for i in range(4,n+1):
                f = sum_queue.popleft()+sum_queue[1]+2 # 递推公式为 f(n) = S_{n-3} + S_{n-1} + 2
                sum_queue.append(f+sum_queue[1])
            return f%(10**9+7)
        
'''更简单的递推式'''
class Solution:
    def numTilings(self, n: int) -> int:
        if n <= 2:
            return n
        elif n == 3:
            return 5
        else:
            dp = [0]*(n+1)
            dp[1],dp[2],dp[3] = 1,2,5
            for i in range(4,n+1):
                dp[i] = (2*dp[i-1]+dp[i-3])%(10**9+7) # 递推公式为 f(n) = 2*f(n-1) + f(n-3)
            return dp[n]

    
'''
官方题解：动态规划
'''
class Solution:
    def numTilings(self, n: int) -> int:
        MOD = 10 ** 9 + 7  # 设置模数，防止结果过大
        # dp[i][j] 表示铺满前i列时，第i+1列状态为j的方案数
        # 状态定义：
        # 0: 第i+1列两行都空
        # 1: 第i+1列第一行空，第二行满
        # 2: 第i+1列第一行满，第二行空
        # 3: 第i+1列两行都满
        dp = [[0] * 4 for _ in range(n + 1)] # [0] * 4 是创建一个包含4个元素的列表，初始值都为0，for _ in range(n + 1) 是创建一个包含 n + 1 个这样的列表的二维列表
        dp[0][3] = 1  # 初始状态：没有列时，认为是完全填满的状态
        
        for i in range(1, n + 1):
            # 状态0：当前列两行都空，只能从前一列完全填满转移而来
            dp[i][0] = dp[i - 1][3]
            
            # 状态1：当前列第一行空，第二行满
            # 可以从两种情况转移：
            # 1. 前一列全空(dp[i-1][0])，加一个L型骨牌
            # 2. 前一列第一行满，第二行空(dp[i-1][2])，加一个横向1×2骨牌
            dp[i][1] = (dp[i - 1][0] + dp[i - 1][2]) % MOD
            
            # 状态2：当前列第一行满，第二行空
            # 可以从两种情况转移：
            # 1. 前一列全空(dp[i-1][0])，加一个L型骨牌（翻转过来）
            # 2. 前一列第一行空，第二行满(dp[i-1][1])，加一个横向1×2骨牌
            dp[i][2] = (dp[i - 1][0] + dp[i - 1][1]) % MOD
            
            # 状态3：当前列两行都满（完全填满）
            # 可以从四种情况转移：
            # 1. 前一列两行都空(dp[i-1][0])，加两个横向1×2骨牌
            # 2. 前一列第一行空，第二行满(dp[i-1][1])，加一个L型骨牌
            # 3. 前一列第一行满，第二行空(dp[i-1][2])，加一个L型骨牌
            # 4. 前一列也完全填满(dp[i-1][3])，加一个2×1骨牌
            dp[i][3] = (((dp[i - 1][0] + dp[i - 1][1]) % MOD + dp[i - 1][2]) % MOD + dp[i - 1][3]) % MOD
        
        # 返回第n列完全填满的方案数
        return dp[n][3]