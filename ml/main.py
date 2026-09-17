import pandas as pd
from src.processor import Processor

import matplotlib
matplotlib.use("Agg")
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
os.makedirs("ml/data", exist_ok=True)

from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

df_sales = pd.read_csv('data/sales.csv')
#regular cleaning
df_sales__wn_rt90 = df_sales[df_sales['route_id'] != 90]
df_sales__wn_cols = df_sales__wn_rt90[['sale_date', 'net_amount', 'profit', 'quantity', 'customer_id', 'product_id']]
df_sales = df_sales__wn_cols

# format datetime
df_sales['sale_date'] = pd.to_datetime(df_sales['sale_date'], format='%Y-%m-%d')

df_agg = df_sales.groupby('sale_date').agg(
    net_amount=('net_amount', 'sum'),
    profit=('profit', 'sum'),
    quantity=('quantity', 'sum'),
    customer_id=('customer_id', 'nunique'),
    product_id=('product_id', 'nunique')
)
df_agg = df_agg.asfreq('D', fill_value=0)
# set zero for records under zero
columns_to_clip = ["net_amount", "profit", "quantity"]
df_agg[columns_to_clip] = df_agg[columns_to_clip].clip(lower=0)

# timeline plot
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(
    data=df_agg,
    y="net_amount",
    x=df_agg.index,
    label="daily net amount over the time",
    color="#1f77b4",
    linewidth=1.2,
    ax=ax,
)
sns.lineplot(
    data=df_agg,
    y="profit",
    x=df_agg.index,
    label="daily profit over the time",
    color="#ff7f0e",
    linewidth=1.2,
    ax=ax,
)

ax.set_title("daily evolution -> net_amount and profit", fontsize=14, pad=10)
ax.set_xlabel("date", fontsize=11)
ax.set_ylabel("amount", fontsize=11)
ax.legend(loc="upper right")
plt.tight_layout()

fig.savefig("data/time_series_sales.png", dpi=300)
plt.close(fig)
print(" temp chart saved in: ml/data/time_series_sales.png")


fig, axes = plt.subplots(1, 3, figsize=(12, 8))
sns.boxplot(
    data=df_agg,
    y="net_amount",
    color="#aec7e8",
    ax=axes[0],
    flierprops={"marker": "o", "color": "red", "alpha": 0.5},
)
axes[0].set_title("daily net amount")
axes[0].set_ylabel("amount")

sns.boxplot(
    data=df_agg,
    y="quantity",
    color="#98df8a",
    ax=axes[1],
    flierprops={"marker": "o", "color": "red", "alpha": 0.5},
)
axes[1].set_title("daily units")
axes[1].set_ylabel("units")

sns.boxplot(
    data=df_agg,
    y="customer_id",
    color="#ffbb78",
    ax=axes[2],
    flierprops={"marker": "o", "color": "red", "alpha": 0.5},
)
axes[2].set_title("daily unique customers")
axes[2].set_ylabel("customers")

plt.tight_layout()
fig.savefig("data/boxplots_outliers.png", dpi=300)
plt.close(fig)
print(" temp boxplot chart saved in: ml/data/boxplots_outliers.png")



# data and feature processor
features = ["net_amount", "profit", "quantity", "customer_id"]
data_raw = df_agg[features].values

train_size = int(len(data_raw) * 0.8)
train_raw = data_raw[:train_size]
test_raw = data_raw[train_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_raw)
test_scaled = scaler.transform(test_raw)

p = Processor()
WINDOW_SIZE = 14
TARGET_COL_IDX = 0

X_train, y_train = p.create_sliding_windows(
    data=train_scaled, target_col_idx=TARGET_COL_IDX, window_size=WINDOW_SIZE
)
X_test, y_test = p.create_sliding_windows(
    data=test_scaled, target_col_idx=TARGET_COL_IDX, window_size=WINDOW_SIZE
)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}")