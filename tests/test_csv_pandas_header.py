import pandas as pd
import os

CSV = "test.csv"

# 1. 先创建一个简单的 CSV，包含表头和两行数据
df0 = pd.DataFrame([
    {"trade_time": 1000, "side": "long",  "price": 50},
    {"trade_time": 1001, "side": "short", "price": 51},
])
df0.to_csv(CSV, index=False)
print("=== 初始 CSV 内容 ===")
print(open(CSV).read())

# 2. 错误地用 header=None, names=… 读入
df_wrong = pd.read_csv(CSV,
    header=None,
    names=["trade_time", "side", "price"]
)
print("=== 错误读取 DataFrame ===")
print(df_wrong)

# 3. 把它覆盖写回 CSV
df_wrong.to_csv(CSV, index=False)
print("=== 覆盖写回后 CSV 内容 ===")
print(open(CSV).read())

# # 4. 正确的读取与写法对比
# # 4.1 重新写回初始内容
# df0.to_csv(CSV, index=False)
# # 4.2 正确读取
# df_ok = pd.read_csv(CSV)  # header 默认就会取第一行
# print("=== 正确读取后 DataFrame ===")
# print(df_ok)
# # 4.3 覆盖写回
# df_ok.to_csv(CSV, index=False)
# print("=== 正确写回后 CSV 内容 ===")
# print(open(CSV).read())

# # 清理
# # os.remove(CSV)