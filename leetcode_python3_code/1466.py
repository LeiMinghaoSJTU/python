'''标准答案'''
from collections import defaultdict, deque
class Solution:
    def minReorder(self, n, connections):
        # 创建邻接表，同时记录边的方向
        graph = defaultdict(list)
        
        # 构建无向图，但标记边的方向
        for a, b in connections:
            # (邻居节点, 是否为原始方向)
            graph[a].append((b, True))   # 原始方向: a -> b
            graph[b].append((a, False))  # 反向: b -> a (虚拟边)
        
        # BFS遍历
        queue = deque([0])  # 从节点0开始
        visited = set([0])
        changes = 0  # 需要改变方向的边数
        
        while queue:
            node = queue.popleft()
            
            # 遍历当前节点的所有邻居
            for neighbor, is_original in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    
                    # 如果这条边是原始方向(away from 0)，需要改变方向
                    if is_original:
                        changes += 1
                    
                    queue.append(neighbor)
        
        return changes
'''这是一个超出时间限制的递归反例'''
# class Solution:
#     def minReorder(self, n: int, connections: List[List[int]]) -> int:
#         def helper(present_city,n,connections):
#             if n==1:
#                 return 0
#             minreorder=0     
#             for i, connection in enumerate(connections):
#                 if connection[0]==present_city:
#                     connections[i]=[-1,-1]
#                     minreorder+=helper(connection[1],n-1,connections)+1
#                 elif connection[1]==present_city:
#                     connections[i]=[-1,-1]
#                     minreorder+=helper(connection[0],n-1,connections)
#             return minreorder
#         return helper(0,n,connections)