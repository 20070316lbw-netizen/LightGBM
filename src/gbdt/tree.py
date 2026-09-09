class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.feature = None   # 分裂特征索引
        self.threshold = None   # 分裂阈值
        self.weight = None   # 叶片点权重(只有叶节点才有)
