import time  # 用于获取当前时间戳
from datetime import datetime, timezone  # datetime 模块和时区支持

# 获取当前 POSIX 时间戳（Unix timestamp，基于 UTC 的秒数）
timestamp = time.time()
print(f"当前时间戳 (Unix timestamp): {timestamp}\n")

# 1. 使用 fromtimestamp 不带 tz 参数：返回本地时区的 naive datetime（取决于系统时区）
local_dt = datetime.fromtimestamp(timestamp)
print("1. fromtimestamp (no tz): 本地时间 (naive)")
print(f"  - 值: {local_dt}")
print(f"  - tzinfo: {local_dt.tzinfo} (None，表示 naive，无时区信息)")
print(f"  - 类型: {type(local_dt)}\n")

# 2. 使用 fromtimestamp 带 tz=timezone.utc 参数：返回 UTC 时区的 aware datetime
utc_aware_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
print("2. fromtimestamp (tz=timezone.utc): UTC 时间 (aware)")
print(f"  - 值: {utc_aware_dt}")
print(f"  - tzinfo: {utc_aware_dt.tzinfo} (UTC，表示 aware，有时区信息)")
print(f"  - 类型: {type(utc_aware_dt)}\n")

# 3. 使用 utcfromtimestamp：返回 UTC 时间，但作为 naive datetime（无 tzinfo）
utc_naive_dt = datetime.utcfromtimestamp(timestamp)
print("3. utcfromtimestamp: UTC 时间 (naive)")
print(f"  - 值: {utc_naive_dt}")
print(f"  - tzinfo: {utc_naive_dt.tzinfo} (None，表示 naive，无时区信息)")
print(f"  - 类型: {type(utc_naive_dt)}\n")

# 对比2和3的值：数值相同，但 aware vs naive 的区别
print("对比2和3：")
print(f"  - 值是否相同? {utc_aware_dt.replace(tzinfo=None) == utc_naive_dt} (是的，数值相同，但 aware 有时区标记)")
print(f"  - 时区偏移 (aware): {utc_aware_dt.utcoffset()} (0，因为是 UTC)")
print(f"  - 注意：如果后续代码涉及时区转换，aware 对象更可靠（避免歧义）。")