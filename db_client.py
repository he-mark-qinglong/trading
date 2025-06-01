# db_client.py

import sqlite3
import pandas as pd
from typing import Optional

class SQLiteWALClient:
    def __init__(self,
                 db_path: str,
                 table: str = "ohlcv",
                 primary_key: str = "ts"):
        """
        db_path: 本地 .db 文件路径
        table:   主表名
        primary_key: 用于去重的主键列名
        """
        self.db_path = db_path
        self.table = table
        self.pk = primary_key
        # 初始化：创建表 & 启用 WAL
        conn = self._connect()
        with conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
              ts    INTEGER PRIMARY KEY,
              open  REAL,
              high  REAL,
              low   REAL,
              close REAL,
              vol   REAL
            );
            """)
        conn.close()

    def _connect(self) -> sqlite3.Connection:
        """返回一个新的连接对象，开启 WAL 并加速设置。"""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=10000;")
        return conn

    def append_df_ignore(self,
                         df: pd.DataFrame,
                         chunksize: int = 5000):
        """
        用 INSERT OR IGNORE 把 df 写入表中，遇重复主键跳过。
        df 必须包含 self.pk 列。
        """
        # 确保 pk 列在 df 中
        if self.pk not in df.columns:
            raise ValueError(f"DataFrame must contain primary key column '{self.pk}'")
        cols = list(df.columns)
        placeholders = ", ".join("?" for _ in cols)
        col_sql = ", ".join(cols)
        sql = (f"INSERT OR IGNORE INTO {self.table} "
               f"({col_sql}) VALUES ({placeholders})")
        data = [tuple(row) for row in df[cols].itertuples(index=False, name=None)]
        conn = self._connect()
        with conn:
            conn.executemany(sql, data)
        conn.close()

    def write_df(self,
                 df: pd.DataFrame,
                 if_exists: str = "append",
                 index: bool = False,
                 **to_sql_kwargs):
        """
        pandas.to_sql 的封装，用于全量覆盖或追加。
        推荐 if_exists='append', method='multi', chunksize=5000。
        """
        conn = self._connect()
        with conn:
            df.to_sql(self.table, conn,
                      if_exists=if_exists,
                      index=index,
                      **to_sql_kwargs)
        conn.close()

    def read_df(self,
                cols: Optional[str] = "*",
                limit: Optional[int] = None,
                order_by: str = "ts ASC") -> pd.DataFrame:
        """
        按指定顺序和条数读取表：
        cols:     要选的列，默认为 '*'
        limit:    最多读取多少条，None 则全表
        order_by: SQL ORDER BY 字句，默认 ts 升序
        """
        sql = f"SELECT {cols} FROM {self.table}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit:
            sql += f" LIMIT {limit}"
        conn = self._connect()
        df = pd.read_sql(sql, conn)
        conn.close()
        return df