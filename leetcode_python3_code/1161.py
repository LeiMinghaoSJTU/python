# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
'''
找到二叉树中每一层节点值的和，返回和最大的那一层的层数（层数从 1 开始）。
如果有多层的节点值之和相同，返回最小的层数。
'''
class Solution:
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        queue = deque([root]) # 双端队列
        max_sum = float('-inf')
        max_level = 0
        current_level = 0

        while queue:
            current_level += 1
            level_size = len(queue) # 当前层节点数
            level_sum = 0 # 当前层节点值的和

            for _ in range(level_size): # _表示不需要使用循环变量
                node = queue.popleft() # 从左侧弹出节点，并赋值给node
                level_sum += node.val # 累加当前层的节点值

                if node.left:
                    queue.append(node.left) # append()方法在队列的右侧添加节点
                if node.right:
                    queue.append(node.right)

            if level_sum > max_sum:
                max_sum = level_sum
                max_level = current_level

        return max_level