# LightGBM

[![CI](https://github.com/20070316lbw-netizen/LightGBM/actions/workflows/ci.yml/badge.svg)](https://github.com/20070316lbw-netizen/LightGBM/actions/workflows/ci.yml)

个人从零实现的 GBDT（梯度提升树）回归器，最初是为了搞懂 LightGBM 背后的
数学原理，现在打包成一个能在其他量化项目里直接 `uv add` 的模型包。

**这个仓库只做一件事：给定训练数据 `(X, y)`，用二阶泰勒展开（梯度 + 海森）
拟合一个回归树集成模型，提供 `fit` / `predict`。** 不做任何数据抓取、
清洗、存储或因子计算——这些留给
[`sources`](https://github.com/20070316lbw-netizen/sources)、
[`load`](https://github.com/20070316lbw-netizen/load)、
[`momfactor`](https://github.com/20070316lbw-netizen/momfactor)，`gbdt`
与它们是平级关系而非上下游，不依赖、不 import 它们中的任何一个；调用方
传入的 `X`/`y` 可以来自任何地方（比如用 `momfactor` 算出来的多个因子拼成
特征矩阵，`y` 是对应的未来收益）。

包名是 `gbdt` 而不是 `lightgbm`——仓库沿用了最初"手写 GBDT 对比真实
LightGBM"这个学习项目的名字，但如果导入名也叫 `lightgbm`，会直接跟同名
的真实依赖撞名（`experiments/compare.py` 里就真的 `import lightgbm as
lgb` 拿来对比效果），所以包名一直是 `gbdt`。

## 安装

作为私有 GitHub 仓库，在其他项目里用 `uv add` 直接从 git 安装：

```bash
# 装最新 main
uv add "git+https://github.com/20070316lbw-netizen/LightGBM.git"

# 推荐: 装一个打好 tag 的版本, 避免上游改动悄悄影响你的项目
uv add "git+https://github.com/20070316lbw-netizen/LightGBM.git@v0.1.0"
```

## 快速开始

```python
import numpy as np
from gbdt import SimpleGBDT

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], dtype=float)
y = np.array([1.2, 4.1, 4.9, 8.3, 10.1, 10.8])

model = SimpleGBDT(n_trees=100, max_depth=3, lr=0.1, min_samples=2)
model.fit(X, y)
pred = model.predict(X)
```

## 模型

| 模块 | 内容 | 备注 |
| --- | --- | --- |
| `gbdt.gbdt` | `SimpleGBDT` | MSE 损失下的梯度提升树: 每轮对当前残差 `g = y_pred - y`（`h` 恒为 1）建一棵树, 叶权重 `w = -G/H`, 分裂点按二阶泰勒展开的 `Gain = 0.5·(G_L²/H_L + G_R²/H_R − (G_L+G_R)²/(H_L+H_R))` 枚举选最大; `n_trees`/`max_depth`/`lr`/`min_samples` 分别控制迭代轮数、树深、学习率、叶节点最小分裂样本数 |
| `gbdt.tree` | `TreeNode` | 树节点: `feature`/`threshold` 是内部节点的分裂条件, `weight` 只在叶节点上有值 |

完整数学推导见 [`notes/`](notes/)（`01_gbdt_math.md` 是从二阶泰勒展开到
分裂增益的完整推导，`02_mse_gradient.md` 专门推 MSE 损失下的梯度/海森）。
`experiments/` 下是拿 `sklearn` 的 diabetes 数据集跟真实 `lightgbm` 对比
效果的脚本（`compare.py`），以及用 `optuna` 调真实 `lightgbm` 参数的脚本
（`best_lgb.py`）——这两个是写这个包时用来验证/参照的学习脚本，不是包对外
的 API，跑的话需要额外装 `lightgbm`/`optuna`/`scikit-learn`（已经在
`dependencies` 里）。`data/gbdt_training_data.csv` 是一份手造的小样本
数据，留作参考，当前没有脚本引用。

## 开发

```bash
uv sync
uv run ruff check .
uv run pytest -v
```

测试全部基于手造的小数组验证梯度、海森、分裂增益、建树、拟合/预测端到端
（包括一个 `lr` 从未生效的历史 bug 的回归测试），不需要真实网络也能跑；
CI（见 `.github/workflows/ci.yml`）在 push/PR 到 `main`/`master` 时会跑
同样这两步。
