# MSE 损失下的梯度推导

## 1. 损失函数定义

$$L(y_i, \hat{y}_i) = \frac{1}{2}(y_i - \hat{y}_i)^2$$

前面的 $\frac{1}{2}$ 是为了求导后消掉系数。

## 2. 一阶梯度推导

用链式法则对 $\hat{y}_i$ 求导：

$$g_i = \frac{\partial L}{\partial \hat{y}_i} = \frac{\partial}{\partial \hat{y}_i} \frac{1}{2}(y_i - \hat{y}_i)^2$$

令 $u = y_i - \hat{y}_i$，则：
- 外层：$\frac{d}{du}\frac{1}{2}u^2 = u$
- 内层：$\frac{\partial u}{\partial \hat{y}_i} = -1$

$$\boxed{g_i = \hat{y}_i - y_i}$$

即预测值减真实值，也就是残差。

## 3. 二阶梯度推导

对 $g_i$ 再求一次导：

$$h_i = \frac{\partial^2 L}{\partial \hat{y}_i^2} = \frac{\partial}{\partial \hat{y}_i}(\hat{y}_i - y_i) = 1$$

$$\boxed{h_i = 1}$$

MSE 下二阶导恒为 1，与预测值无关。

## 4. 代入叶权重公式

MSE 下 $H_j = \sum_{i \in j} 1 = n_j$（叶内样本数），因此：

$$w_j^* = -\frac{G_j}{H_j} = -\frac{\sum_{i \in j}(\hat{y}_i - y_i)}{n_j}$$

**直觉：叶节点输出值 = 该叶内残差均值取负 = 平均还差多少，补回去。**

## 5. 数值示例

| 样本 | $y_i$ | $\hat{y}_i$ | $g_i = \hat{y}_i - y_i$ |
|------|--------|--------------|--------------------------|
| 1    | 10     | 7            | -3                       |
| 2    | 5      | 3            | -2                       |
| 3    | 8      | 6            | -2                       |

$$G_j = -7, \quad H_j = 3$$

$$w_j^* = -\frac{-7}{3} = +2.33$$

这棵树在此叶节点输出 +2.33，加到当前预测值上，让预测更接近真实值。