# GBDT 完整数学推导

## 1. 问题定义

给定训练数据 $(x_i, y_i)$，当前预测值为 $\hat{y}_i$，目标是学习一棵树 $f(x)$ 使得：

$$\hat{y}_i \rightarrow \hat{y}_i + f(x_i)$$

## 2. Boosting 迭代框架

$$\hat{y}^{(0)} = 0$$
$$\hat{y}^{(1)} = \hat{y}^{(0)} + f_1(x)$$
$$\hat{y}^{(2)} = \hat{y}^{(1)} + f_2(x)$$
$$\vdots$$
$$\hat{y}^{(t)} = \hat{y}^{(t-1)} + f_t(x)$$

每一轮加一棵树，专门拟合当前的残差。

## 3. 二阶泰勒展开

对损失函数 $L(y_i, \hat{y}_i + f(x_i))$ 在 $\hat{y}_i$ 处展开：

$$L(y_i, \hat{y}_i + f(x_i)) \approx L(y_i, \hat{y}_i) + g_i f(x_i) + \frac{1}{2} h_i f(x_i)^2$$

其中：
$$g_i = \frac{\partial L}{\partial \hat{y}_i}, \quad h_i = \frac{\partial^2 L}{\partial \hat{y}_i^2}$$

忽略常数项，本轮需要最小化的目标为：

$$\mathcal{L} = \sum_i \left( g_i f(x_i) + \frac{1}{2} h_i f(x_i)^2 \right)$$

## 4. 叶节点参数化

假设树有 $T$ 个叶节点，同一叶节点 $j$ 内所有样本输出相同的值 $w_j$，即 $f(x_i) = w_j$（$i \in j$）。

定义：
$$G_j = \sum_{i \in j} g_i, \quad H_j = \sum_{i \in j} h_i$$

目标函数化简为：

$$\mathcal{L} = \sum_j \left( G_j w_j + \frac{1}{2} H_j w_j^2 \right)$$

## 5. 最优叶权重

对每个叶节点的 $w_j$ 求导，令其为 0：

$$\frac{\partial \mathcal{L}}{\partial w_j} = G_j + H_j w_j = 0$$

$$\boxed{w_j^* = -\frac{G_j}{H_j}}$$

代入得最优得分：

$$\text{Score}_j = -\frac{G_j^2}{2H_j}$$

## 6. 分裂增益推导

考虑将一个叶节点分裂成左右两个子节点：

- 父节点得分：$-\dfrac{(G_L+G_R)^2}{2(H_L+H_R)}$
- 左子节点得分：$-\dfrac{G_L^2}{2H_L}$
- 右子节点得分：$-\dfrac{G_R^2}{2H_R}$

分裂收益 = 分裂前损失 − 分裂后损失：

$$\boxed{\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L} + \frac{G_R^2}{H_R} - \frac{(G_L+G_R)^2}{H_L+H_R}\right]}$$

遍历所有特征和阈值，选择 Gain 最大的分裂点。若 Gain $\leq 0$，则不分裂。