# 加载数据，检查数据的基本信息和分布
# 确保数据加载正确，识别可能需要处理的缺失值或异常值

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


# 加载数据
df = pd.read_csv("data/customer_booking.csv", encoding="ISO-8859-1")

# 初步探索
#print(df.head())
#print(df.info())
#print(df["flight_day"].unique())

# 映射 flight_day
mapping = {
    "Mon": 1,
    "Tue": 2,
    "Wed": 3,
    "Thu": 4,
    "Fri": 5,
    "Sat": 6,
    "Sun": 7,
}
df["flight_day"] = df["flight_day"].map(mapping)

# One-Hot Encoding 处理非数值data
df_encoded = pd.get_dummies(df, columns=['route', 'booking_origin', 'sales_channel', 'trip_type'], drop_first=True)

# 保存预处理后的数据
df.to_csv("data/processed2_customer_booking.csv", index=False)
print("Data saved successfully.")

# 将数据划分为训练集合测试集，确保模型可以被评估

from sklearn.model_selection import train_test_split

# 定义特征变量 X 和目标变量 y
X = df.drop('booking_complete', axis=1)  # 特征
y = df['booking_complete']  # 目标变量

# 对整个数据集进行 One-Hot 编码
X_encoded = pd.get_dummies(X, columns=['sales_channel', 'trip_type', 'route', 'booking_origin'], drop_first=True)

# 再次划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# 确认非数值列
print(X_train.dtypes)


# 训练逻辑回归模型

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. 特征缩放
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 确认编码和缩放后，所有特征都是数值形式的
print(X_train_scaled[:5])  # 查看前5行缩放后的特征


# 3. 训练模型
model = LogisticRegression(class_weight='balanced', random_state=42, solver='newton-cg', max_iter=500)
model.fit(X_train_scaled, y_train)


# 评估模型在测试集上的表现

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# 预测，评估模型在测试集上的表现
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]  # 获取正类的概率

# 打印分类报告
print('Classification Report:\n', classification_report(y_test, y_pred))

# 计算 AUC
print('AUC:', roc_auc_score(y_test, y_proba))

# 打印混淆矩阵
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

import seaborn as sns

# 计算正确分类和错误分类的数量
confusion_results = results_df.groupby(['Actual', 'Correct']).size().unstack(fill_value=0)

# 可视化
confusion_results.plot(kind='bar', stacked=True, figsize=(10, 6), color=['red', 'green'])
plt.xlabel('Actual Class')
plt.ylabel('Count')
plt.title('Correct vs Incorrect Predictions by Class')
plt.legend(['Incorrect', 'Correct'])
plt.show()

# 1. 查看模型系数。通过逻辑回归的系数了解每个特征对目标变量的影响
import numpy as np

# 查看系数和特征名称
#coefficients = model.coef_[0]
features = X_encoded.columns

# 提取特征重要性
importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

# 打印结果
print(importance)

# 2. 可视化特征重要性。用柱状图展示。

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
importance.set_index('Feature').plot(kind='barh', legend=False)
#plt.barh(importance['Feature'], importance['Coefficient'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.title('Feature Importance')
plt.show()