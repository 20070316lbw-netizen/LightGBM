from gbdt.tree import TreeNode


def test_tree_node_defaults_are_none():
    node = TreeNode()
    assert node.left is None
    assert node.right is None
    assert node.feature is None
    assert node.threshold is None
    assert node.weight is None


def test_tree_node_fields_are_independent():
    left = TreeNode()
    right = TreeNode()
    node = TreeNode()
    node.left = left
    node.right = right
    node.feature = 2
    node.threshold = 1.5

    assert node.left is left
    assert node.right is right
    assert node.weight is None  # 内部节点没有 weight
