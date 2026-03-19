import numpy as np

class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.feature = None   # 分裂特征索引
        self.threshold = None   # 分裂阈值
        self.weight = None   # 叶片点权重(只有叶节点才有)

class SimpleGBDT:
    def __init__(self, n_trees=10, max_depth=3, lr=0.1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.lr = lr
        self.trees = []

    def _grad(self, y, y_pred):
        return y_pred - y   # 这是 g_i
    

    def _hess(self, y, y_pred):
        return np.ones_like(y)   # h_i = 1

        y_pred = np.zeros_like(y)
        node = TreeNode()
        
        for _ in range(self.n_trees):
            g = self._grad(y, y_pred)
            h = self._hess(y, y_pred)



if __name__ == "__main__":
    X = np.array([1,2],
                 [3,4],
                 [5,6],
                 [7,8],
                 [9,10])