import numpy as np
import pytest

from gbdt.gbdt import SimpleGBDT

# ---------------------------------------------------------------------------
# 梯度 / 二阶导（MSE 损失）
# ---------------------------------------------------------------------------

def test_grad_is_residual():
    model = SimpleGBDT()
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 1.5, 1.5])
    np.testing.assert_allclose(model._grad(y, y_pred), y_pred - y)


def test_hess_is_all_ones():
    model = SimpleGBDT()
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(model._hess(y, y_pred), np.ones_like(y))


# ---------------------------------------------------------------------------
# 分裂增益
# ---------------------------------------------------------------------------

def test_gain_is_zero_when_either_side_empty():
    model = SimpleGBDT()
    assert model._gain(G_L=1.0, H_L=0.0, G_R=2.0, H_R=3.0) == 0
    assert model._gain(G_L=1.0, H_L=2.0, G_R=2.0, H_R=0.0) == 0


def test_gain_matches_closed_form():
    model = SimpleGBDT()
    G_L, H_L, G_R, H_R = -4.0, 2.0, 6.0, 3.0
    expected = 0.5 * (G_L**2 / H_L + G_R**2 / H_R - (G_L + G_R) ** 2 / (H_L + H_R))
    assert model._gain(G_L, H_L, G_R, H_R) == pytest.approx(expected)


def test_gain_is_non_negative_for_a_real_split():
    # 分裂几乎总是不会让损失变差（凸性保证），这里挑一组不对称的数字验证
    model = SimpleGBDT()
    assert model._gain(G_L=-5.0, H_L=2.0, G_R=5.0, H_R=2.0) >= 0


# ---------------------------------------------------------------------------
# 最优分裂点搜索
# ---------------------------------------------------------------------------

def test_best_split_finds_the_obvious_threshold():
    model = SimpleGBDT()
    # 单特征，真实分界在 x=3 附近：x<=3 对应低梯度组，x>3 对应高梯度组
    X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
    g = np.array([-3.0, -3.0, -3.0, 3.0, 3.0, 3.0])
    h = np.ones_like(g)

    feature, threshold, gain = model._best_split(X, g, h)

    assert feature == 0
    assert threshold == 3
    assert gain > 0


# ---------------------------------------------------------------------------
# 建树
# ---------------------------------------------------------------------------

def test_build_tree_leaf_when_too_few_samples():
    model = SimpleGBDT(min_samples=5)
    X = np.array([[1.0], [2.0]])
    g = np.array([2.0, 4.0])
    h = np.array([1.0, 1.0])

    node = model._build_tree(X, g, h, depth=0)

    assert node.feature is None
    assert node.weight == pytest.approx(-g.sum() / h.sum())


def test_build_tree_leaf_when_max_depth_reached():
    model = SimpleGBDT(max_depth=0)
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    g = np.array([1.0, -1.0, 1.0, -1.0])
    h = np.ones_like(g)

    node = model._build_tree(X, g, h, depth=0)

    assert node.feature is None
    assert node.weight == pytest.approx(-g.sum() / h.sum())


def test_build_tree_splits_when_gain_is_positive():
    model = SimpleGBDT(max_depth=3, min_samples=1)
    X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
    g = np.array([-3.0, -3.0, -3.0, 3.0, 3.0, 3.0])
    h = np.ones_like(g)

    node = model._build_tree(X, g, h, depth=0)

    assert node.feature == 0
    assert node.left is not None
    assert node.right is not None


# ---------------------------------------------------------------------------
# L2 正则(lambda_l2)
# ---------------------------------------------------------------------------

def test_leaf_weight_without_l2_matches_original_formula():
    model = SimpleGBDT(lambda_l2=0.0)
    g = np.array([1.0, 2.0, -1.0])
    h = np.array([1.0, 1.0, 1.0])
    assert model._leaf_weight(g, h) == pytest.approx(-g.sum() / h.sum())


def test_leaf_weight_matches_l2_closed_form():
    model = SimpleGBDT(lambda_l2=2.0)
    g = np.array([1.0, 2.0, -1.0])
    h = np.array([1.0, 1.0, 1.0])
    expected = -g.sum() / (h.sum() + 2.0)
    assert model._leaf_weight(g, h) == pytest.approx(expected)


def test_larger_l2_shrinks_leaf_weight_towards_zero():
    g = np.array([-4.0, -4.0])
    h = np.array([1.0, 1.0])

    no_reg = SimpleGBDT(lambda_l2=0.0)
    with_reg = SimpleGBDT(lambda_l2=10.0)

    assert abs(with_reg._leaf_weight(g, h)) < abs(no_reg._leaf_weight(g, h))


def test_gain_matches_l2_closed_form():
    lam = 1.5
    model = SimpleGBDT(lambda_l2=lam)
    G_L, H_L, G_R, H_R = -4.0, 2.0, 6.0, 3.0
    expected = 0.5 * (
        G_L**2 / (H_L + lam)
        + G_R**2 / (H_R + lam)
        - (G_L + G_R) ** 2 / (H_L + H_R + lam)
    )
    assert model._gain(G_L, H_L, G_R, H_R) == pytest.approx(expected)


def test_gain_is_zero_when_either_side_empty_with_l2():
    # lambda_l2 参与除零保护: H+lambda_l2 == 0 才算"空", 而不是 H == 0
    model = SimpleGBDT(lambda_l2=2.0)
    assert model._gain(G_L=1.0, H_L=-2.0, G_R=2.0, H_R=3.0) == 0
    assert model._gain(G_L=1.0, H_L=2.0, G_R=2.0, H_R=-2.0) == 0


def test_build_tree_leaf_weight_uses_lambda_l2():
    model = SimpleGBDT(min_samples=5, lambda_l2=4.0)
    X = np.array([[1.0], [2.0]])
    g = np.array([2.0, 4.0])
    h = np.array([1.0, 1.0])

    node = model._build_tree(X, g, h, depth=0)

    assert node.weight == pytest.approx(-g.sum() / (h.sum() + 4.0))


def test_fit_with_stronger_l2_increases_train_error(toy_data):
    # 更大的 lambda_l2 压缩叶权重, 拟合能力变弱, 训练集误差应该上升(但仍能正常出树)
    X, y = toy_data
    no_reg = SimpleGBDT(n_trees=10, max_depth=3, lr=0.3, min_samples=1, lambda_l2=0.0)
    no_reg.fit(X, y)

    reg = SimpleGBDT(n_trees=10, max_depth=3, lr=0.3, min_samples=1, lambda_l2=5.0)
    reg.fit(X, y)

    assert len(reg.trees) == 10
    mse_no_reg = np.mean((no_reg.predict(X) - y) ** 2)
    mse_reg = np.mean((reg.predict(X) - y) ** 2)
    assert mse_reg > mse_no_reg


# ---------------------------------------------------------------------------
# fit / predict 端到端，以及针对之前那个 lr 从未生效 bug 的回归测试
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_data():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], dtype=float)
    y = np.array([1.2, 4.1, 4.9, 8.3, 10.1, 10.8])
    return X, y


def test_fit_predict_reduces_error_versus_zero_baseline(toy_data):
    X, y = toy_data
    model = SimpleGBDT(n_trees=20, max_depth=3, lr=0.3, min_samples=1)
    model.fit(X, y)
    pred = model.predict(X)

    baseline_mse = np.mean(y**2)  # 全 0 预测的 MSE
    model_mse = np.mean((pred - y) ** 2)
    assert model_mse < baseline_mse


def test_learning_rate_actually_scales_predictions(toy_data):
    # 回归测试：修复前 lr 从未在 fit/predict 里生效，
    # 这里验证不同 lr 会产生不同大小的预测（且大致成比例）。
    X, y = toy_data

    small_lr = SimpleGBDT(n_trees=1, max_depth=2, lr=0.1, min_samples=1)
    small_lr.fit(X, y)
    pred_small = small_lr.predict(X)

    big_lr = SimpleGBDT(n_trees=1, max_depth=2, lr=0.5, min_samples=1)
    big_lr.fit(X, y)
    pred_big = big_lr.predict(X)

    assert not np.allclose(pred_small, pred_big)
    # 单棵树、同样的分裂结构下，预测应正比于 lr
    ratio = pred_big / pred_small
    np.testing.assert_allclose(ratio, np.full_like(ratio, 5.0), rtol=1e-6)


def test_more_trees_keep_improving_fit(toy_data):
    X, y = toy_data

    few_trees = SimpleGBDT(n_trees=1, max_depth=3, lr=0.3, min_samples=1)
    few_trees.fit(X, y)
    mse_few = np.mean((few_trees.predict(X) - y) ** 2)

    many_trees = SimpleGBDT(n_trees=30, max_depth=3, lr=0.3, min_samples=1)
    many_trees.fit(X, y)
    mse_many = np.mean((many_trees.predict(X) - y) ** 2)

    assert mse_many < mse_few


# ---------------------------------------------------------------------------
# 早停(early stopping)
# ---------------------------------------------------------------------------

@pytest.fixture
def val_split_data():
    # 简单的线性关系, 划出独立的训练/验证集
    rng = np.random.RandomState(0)
    X = rng.rand(60, 2)
    y = X[:, 0] * 3 - X[:, 1] * 2
    return X[:40], y[:40], X[40:], y[40:]


def test_fit_without_validation_data_keeps_all_trees_and_no_best_iteration(toy_data):
    X, y = toy_data
    model = SimpleGBDT(n_trees=5, max_depth=2, lr=0.3, min_samples=1)
    model.fit(X, y)

    assert len(model.trees) == 5
    assert model.best_iteration is None
    assert model.best_val_loss is None


def test_fit_with_validation_data_tracks_best_iteration(val_split_data):
    X_train, y_train, X_val, y_val = val_split_data
    model = SimpleGBDT(n_trees=30, max_depth=3, lr=0.3, min_samples=1)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    assert model.best_iteration is not None
    assert 1 <= model.best_iteration <= 30
    assert model.best_val_loss >= 0 # type: ignore
    # 即使没有触发早停, 也会回滚到验证集最优的那一轮
    assert len(model.trees) == model.best_iteration


def test_early_stopping_rounds_stops_training_before_n_trees(val_split_data):
    X_train, y_train, X_val, y_val = val_split_data
    model = SimpleGBDT(n_trees=200, max_depth=6, lr=0.5, min_samples=1)
    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        early_stopping_rounds=3,
    )

    assert len(model.trees) < 200
    assert len(model.trees) == model.best_iteration


def test_rolled_back_model_matches_recorded_best_val_loss(val_split_data):
    X_train, y_train, X_val, y_val = val_split_data
    model = SimpleGBDT(n_trees=50, max_depth=4, lr=0.5, min_samples=1)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, early_stopping_rounds=5)

    # 回滚后的模型在验证集上重新预测算出来的 MSE, 应该正好等于训练时记录的 best_val_loss
    recomputed_val_loss = np.mean((y_val - model.predict(X_val)) ** 2)
    assert recomputed_val_loss == pytest.approx(model.best_val_loss)
