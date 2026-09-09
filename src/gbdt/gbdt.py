import numpy as np

from gbdt.tree import TreeNode


class SimpleGBDT:
    def __init__(self, n_trees=100, max_depth=5, lr=0.1, min_samples=2, lambda_l2=0.0):
        self.n_trees = n_trees   # boosting 轮数,即最终有多少棵树
        self.max_depth = max_depth   # 单棵树的最大深度
        self.lr = lr   # 学习率,每棵树的输出乘以它再累加到预测值上
        self.trees = []   # 已训练好的树,按顺序存放,predict 时依次遍历累加
        self.min_samples = min_samples   # 节点样本数低于这个值就不再分裂,直接变叶子
        self.lambda_l2 = lambda_l2   # 叶权重的 L2 正则系数,越大叶权重越被压向 0
        self.best_iteration = None   # 早停后回滚到的最优轮数(树的数量)
        self.best_val_loss = None   # 早停过程中验证集上出现过的最小 loss

    def _grad(self, y, y_pred):
        return y_pred - y   # 这是 g_i

    def _hess(self, y, y_pred):
        return np.ones_like(y)   # h_i = 1

    def _leaf_weight(self, g, h):
        # 叶权重公式 w = -G/(H+lambda_l2);lambda_l2=0 时退化为无正则的 -G/H
        return -g.sum() / (h.sum() + self.lambda_l2)

    def _gain(self, G_L, H_L, G_R, H_R):   # 这里填公式
        # 防止除以0
        if H_L + self.lambda_l2 == 0 or H_R + self.lambda_l2 == 0:
            return 0

        Gain = 0.5 * (
            G_L**2 / (H_L + self.lambda_l2)
            + G_R**2 / (H_R + self.lambda_l2)
            - (G_L + G_R) ** 2 / (H_L + H_R + self.lambda_l2)
        )
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
        best_gain = -np.inf   # 定义 Gain 初始化最大收益为负无穷.
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            # 取出这一列的值,排序去重
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                # 定义左右两组
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                # 计算 G_L, H_L, G_R, H_R
                G_L, H_L = g[left_mask].sum(), h[left_mask].sum()
                G_R, H_R = g[right_mask].sum(), h[right_mask].sum()

                # Gain 我已经在 def _gain 里面返回过了,这里我只需要比较和记录就好
                gain = self._gain(G_L, H_L, G_R, H_R)
                if gain > best_gain:
                    best_gain = gain    # 更新最大gain
                    best_feature = feature_idx    # 记录是哪个特征
                    best_threshold = threshold    # 记录是哪个阈值

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, g, h, depth):   # 递归建树
        # 1. 如果样本太少：→ 创建叶节点，weight = -g.sum() / (h.sum()+lambda_l2)，return
        if len(g) < self.min_samples:
            node = TreeNode()
            node.weight = self._leaf_weight(g, h)
            return node

        # 2. 如果深度达到上限：→ 创建叶节点，weight = -g.sum() / (h.sum()+lambda_l2)，return
        if depth >= self.max_depth:
            node = TreeNode()
            node.weight = self._leaf_weight(g, h)
            return node

        # 3. 找最优分裂点
        feature, threshold, best_gain = self._best_split(X, g, h)

        # 4. 如果 best_gain <= 0： → 创建叶节点，weight = -g.sum() / (h.sum()+lambda_l2)，return
        if best_gain <= 0:
            node = TreeNode()
            node.weight = self._leaf_weight(g, h)
            return node

        # 5. 用 mask 把样本分成左右两组
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        X_left, X_right = X[left_mask], X[right_mask]
        g_left, g_right = g[left_mask], g[right_mask]
        h_left, h_right = h[left_mask], h[right_mask]

        # 6. 递归建左右子树
        node = TreeNode()
        node.feature = feature # type: ignore
        node.threshold = threshold
        node.left = self._build_tree(X_left, g_left, h_left, depth + 1) # type: ignore
        node.right = self._build_tree(X_right, g_right, h_right, depth + 1) # type: ignore
        return node

    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=None):
        """
        1. 初始化预测值 y_pred = 全0
        2. 循环 n_trees 次：
            a. 算 g 和 h
            b. 建一棵树
            c. 把树存进 self.trees
            d. 用 lr 缩放后更新 y_pred
        3. 如果给了验证集(X_val/y_val)：每轮结束后算一次验证集 MSE，
           记录历史最优轮数；若连续 early_stopping_rounds 轮没有改善就停止，
           最后把 self.trees 回滚到最优轮数,避免带着过拟合的树。
        """
        y_pred = np.zeros_like(y, dtype=float)

        use_val = X_val is not None and y_val is not None
        if use_val:
            y_val_pred = np.zeros_like(y_val, dtype=float)
            rounds_no_improve = 0

        for i in range(self.n_trees):
            g = self._grad(y, y_pred)
            h = self._hess(y, y_pred)
            tree = self._build_tree(X, g, h, depth=0)
            self.trees.append(tree)
            y_pred += self.lr * np.array([self._predict_single(tree, x) for x in X])

            if use_val:
                y_val_pred += self.lr * np.array([self._predict_single(tree, x) for x in X_val])
                val_loss = np.mean((y_val - y_val_pred) ** 2)

                if self.best_val_loss is None or val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_iteration = i + 1
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1
                    stop_now = (
                        early_stopping_rounds is not None
                        and rounds_no_improve >= early_stopping_rounds
                    )
                    if stop_now:
                        break

        if use_val:
            # 回滚到验证集上表现最好的那一轮，丢弃之后过拟合的树
            self.trees = self.trees[: self.best_iteration]

    def predict(self, X):
        # 每棵树各自预测一个值,乘以学习率后逐棵累加,得到最终预测(与 fit 里更新 y_pred 的方式一致)
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            for i, x in enumerate(X):
                y_pred[i] += self.lr * self._predict_single(tree, x)
        return y_pred

    def _predict_single(self, node, x):
        # 从根节点递归往下走:走到叶节点(weight 不为 None)就返回该叶的权重,
        # 否则按 feature/threshold 决定往左还是往右子树继续走
        if node.weight is not None:
            return node.weight
        else:
            if x[node.feature] <= node.threshold:
                return self._predict_single(node.left, x)
            else:
                return self._predict_single(node.right, x)


if __name__ == "__main__":
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8],
                  [9, 10],
                  [11, 12]])
    y = np.array([1.2, 4.1, 4.9, 8.3, 10.1, 10.8])

    model = SimpleGBDT(n_trees=5, max_depth=3, lr=0.1)
    model.fit(X, y)
    pred = model.predict(X)
    print("预测值:", pred)
    print("真实值:", y)
