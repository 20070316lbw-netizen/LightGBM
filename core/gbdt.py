import numpy as np
from core.tree import TreeNode

class SimpleGBDT:
    def __init__(self, n_trees=100, max_depth=5, lr=0.1, min_samples=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.lr = lr
        self.trees = []
        self.min_samples = min_samples


    def _grad(self, y, y_pred):
        return y_pred - y   # 这是 g_i
    

    def _hess(self, y, y_pred):
        return np.ones_like(y)   # h_i = 1
    

    def _gain(self, G_L, H_L, G_R, H_R):   # 这里填公式
        #防止除以0
        if H_L == 0 or H_R == 0:
            return 0

        Gain =  0.5 * (G_L**2/H_L + G_R**2/H_R - (G_L+G_R)**2/(H_L+H_R))
        return Gain
    

    def _best_split(self, X, g, h):   # 枚举所有特征和阈值,找出最大 Gain 的分裂点
        """
        思路是遍历该特征:
            遍历该特征的每个可能阈值:
                把样本分成左右两组
                计算 G_L, H_L, G_R, H_R
                计算 Gain
                如果 Gain > 当前最大值 Gain,就记录下来
        返回最值 (feature, threshould)
        """
        """
        best_gain
        best_feature 
        best_threshold 
        这三个变量是用来记录目前为止最好的分裂点的,每次算出一个新的 gain，就和 best_gain 比较
        """
        best_gain = -np.inf   # 定义 Gain 初始化最大收益为负无穷.
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            # 取出这一列的值,排序去重
            thresholds = np.unique(X[:, feature_idx])
            # 我在 self.__init__ 写过 threshold 了,不要重复写

            for threshold in thresholds:
                # 定义左右两组
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                # 计算 G_L, H_L, G_R, H_R
                G_L, H_L = g[left_mask].sum(), h[left_mask].sum()
                G_R, H_R = g[right_mask].sum(),h[right_mask].sum()

                # Gain 我已经在 def _gain 里面返回过了,这里我只需要比较和记录就好
                gain = self._gain(G_L, H_L, G_R, H_R)
                if gain > best_gain:
                        best_gain = gain    # 更新最大gain
                        best_feature = feature_idx    # 记录是哪个特征
                        best_threshold = threshold    # 记录是哪个阈值

        return best_feature, best_threshold, best_gain


    def _build_tree(self, X, g, h, depth):   # 递归建树
        # 输入：X（特征）, g（一阶梯度）, h（二阶梯度）, depth（当前深度）
        # 1. 如果样本太少：→ 创建叶节点，weight = -g.sum() / h.sum()，return
        if len(g) < self.min_samples:
            node = TreeNode()
            node.weight = -g.sum() / h.sum()
            return node

        # 2. 如果深度达到上限：→ 创建叶节点，weight = -g.sum() / h.sum()，return
        if depth >= self.max_depth:
            node = TreeNode()
            node.weight = -g.sum() / h.sum()
            return node

        # 3. 找最优分裂点：→ feature, threshold, best_gain = self._best_split(X, g, h)
        # feature → 最优分裂用的是哪个特征（比如第0列还是第1列）
        # threshold → 最优分裂的阈值是多少（比如 <= 3.0）
        # best_gain → 这次分裂能带来多少收益
        feature, threshold, best_gain = self._best_split(X, g, h)

        # 4. 如果 best_gain <= 0： → 创建叶节点，weight = -g.sum() / h.sum()，return
        if best_gain <= 0:
            node = TreeNode()
            node.weight = -g.sum() / h.sum()
            return node
        # 5. 用 mask 把样本分成左右两组
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        X_left, X_right = X[left_mask], X[right_mask]
        g_left, g_right = g[left_mask], g[right_mask]
        h_left, h_right = h[left_mask], h[right_mask]

        # 6. 递归建左子树：→ node.left = self._build_tree(X_left, g_left, h_left, depth+1)
        node = TreeNode()
        node.feature = feature
        node.threshold = threshold
        node.left = self._build_tree(X_left, g_left, h_left, depth+1)
        node.right = self._build_tree(X_right, g_right, h_right, depth+1)
        return node


    def fit(self, X, y):
        """
        1. 初始化预测值 y_pred = 全0
        2. 循环 n_trees 次：
            a. 算 g 和 h
            b. 建一棵树
            c. 把树存进 self.trees
            d. 更新 y_pred
        """
        y_pred = np.zeros_like(y)
        
        
        
        for _ in range(self.n_trees):
            g = self._grad(y, y_pred)
            h = self._hess(y, y_pred)
            tree = self._build_tree(X, g, h, depth=0)
            self.trees.append(tree)
            y_pred += np.array([self._predict_single(tree, x) for x in X])

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            for i, x in enumerate(X):
                y_pred[i] += self._predict_single(tree, x)
        return y_pred 
    

    def _predict_single(self, node, x):
        if node.weight is not None:
                return node.weight
        else:
            if x[node.feature] <= node.threshold:
                    return self._predict_single(node.left, x)
            else:
                    return self._predict_single(node.right, x)

        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            for i, x in enumerate(X):
                y_pred[i] += self.lr * self._predict_single(tree, x)
        return y_pred
            