
import pandas as pd
import datetime

def resample_to_7_5m(df):
    df = df.set_index('datetime', drop=True)
    # 创建一个空的列表以存储合并结果
    resampled_data = []

    # 将 DataFrame 以 15 分钟为单位进行重采样
    # for ts, group in df.resample('15min'):
    for ts, group in df.resample('7.5min'):
        if not group.empty:  # 确保组中有数据
            open_price = group.iloc[0]['open']  # 第一条数据的开盘价
            high_price = group['high'].max()     # 最高价
            low_price = group['low'].min()       # 最低价
            close_price = group.iloc[-1]['close']  # 最后一条数据的收盘价
            volume = group['vol'].sum()          # 成交量总和
            
            # 转换时间戳为可读的字符串形式
            datetime_str = ts.strftime('%Y-%m-%d %H:%M:%S')

            # 将合并的数据记录到列表中
            resampled_data.append({
                'datetime': datetime_str,  # 转换为字符串的时间
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'vol': volume
            })

    # 将合并的结果转换为 DataFrame，设置 ts 为索引
    if resampled_data:
        resampled_df = pd.DataFrame(resampled_data)
        # 将 ts 列作为索引
        resampled_df['ts'] = [pd.Timestamp(dt['datetime']).timestamp() for dt in resampled_data]
        resampled_df.set_index('ts', inplace=True)
    else:
        # 如果没有数据，返回空的 DataFrame
        resampled_df = pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'vol'])

    # print(resampled_df.head(5))
    return resampled_df


def read_and_sort_df(client, LIMIT_K_N):
    df = client.read_df(limit=LIMIT_K_N, order_by="ts DESC")
    #print('df.head:', df.head)
    # 2. 检查必须列
    required = {"ts","open","high","low","close","vol"}
    if not required.issubset(df.columns):
        miss = required - set(df.columns)
    # 3. 转换时间
    df["ts"] = df["ts"].astype(int)
    df = df.drop_duplicates("ts").sort_values("ts")
    df["datetime"] =  pd.to_datetime(df["ts"], unit="s")
    df = df.set_index("ts", drop=True)

    # 6) 保证数据是连续、升序的
    df = df.sort_index()
    # print(df.head(5))
    return resample_to_7_5m(df)
    return df