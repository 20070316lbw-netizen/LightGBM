import numpy as np
import lightgbm as lgb
from core import SimpleGBDT
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 读数据
data = load_diabetes()
X, y = data.data, data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 我们的模型
our_model = SimpleGBDT(n_trees=100, max_depth=3, lr=0.1)
our_model.fit(X_train, y_train)
our_pred = our_model.predict(X_test)

# lightgbm库
lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)

print("我们的MSE:  ", round(mean_squared_error(y_test, our_pred), 2))
print("lightgbm MSE:", round(mean_squared_error(y_test, lgb_pred), 2))