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
        best_gain = -np.inf   # 定义 Gain 初始化最大收益为负无穷
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            # 取出这一列的值,排序去重
            thresholds = np.unique(X[:, feature_idx])
            # 我在 self.__init__ 写过 threshold 了,不要重复写

            for threshold in thresholds:
                # 定义左右两组
                left_mask = X[:, feature_idx]
                right_mask = ~left_mask

                # 计算 G_L, H_L, G_R, H_R
                G_L, H_L = g[left_mask].sum(), h[left_mask].sum()
                G_R, H_R = g[right_mask].sum(),h[right_mask].sum()


                # Gain 我已经在 def _gain 里面返回过了



    def _build_tree(self, X, g, h, depth):   # 递归建树
        pass


    def fit(self, X, y):
        pass


    def predict(self, X):
        pass







