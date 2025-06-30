
import os
import time
import matplotlib  
matplotlib.use('Agg')  # 无GUI后端，适合生成图像文件，不显示窗口  
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import numpy as np

def plot_all_multiftfpoc_vars(multFramevp_poc,
                              symbol='',
                              is_trading=False,
                              save_to_file=True):
    # —— 开头：确保所有 index 都是 DatetimeIndex —— 
    df = getattr(multFramevp_poc, 'df', None)
    if isinstance(df, pd.DataFrame):
        # 如果没有 datetime 列，就把原 ts 索引（int 秒）转换
        if not isinstance(df.index, pd.DatetimeIndex):
            # 假设原 index 是 unix timestamp (s)
            df = df.copy()
            df.index = pd.to_datetime(df.index.astype(int), unit='s')
        # 如果你还有单独的 datetime 列，也可以更新一下
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        multFramevp_poc.df = df

    # 其它 series 也同理
    vars_to_plot = [
      'LFrame_vwap_poc','SFrame_vwap_poc', 'SFrame_vwap_up_poc',
      'SFrame_vwap_up_sl','SFrame_vwap_down_poc','SFrame_vwap_down_sl',
      'HFrame_vwap_up_getin','HFrame_vwap_up_sl','HFrame_vwap_up_sl2',
      'HFrame_vwap_down_getin','HFrame_vwap_down_sl','HFrame_vwap_down_sl2',
    ]
    for var in vars_to_plot:
        s = getattr(multFramevp_poc, var, None)
        if isinstance(s, pd.Series) and not isinstance(s.index, pd.DatetimeIndex):
            setattr(multFramevp_poc, var, 
                    pd.Series(s.values,
                              index=pd.to_datetime(s.index.astype(int), unit='s'),
                              name=s.name))

    # ———— 开始画图 ————
    fig, (ax_k, ax_vol) = plt.subplots(
        2,1, figsize=(15,8), sharex=True,
        gridspec_kw={'height_ratios':[7,3]},
        constrained_layout=True
    )
    fig.patch.set_facecolor('black')
    ax_k.set_facecolor('black')
    ax_vol.set_facecolor('black')

    # 1) K 线 / close
    if isinstance(df, pd.DataFrame) and {'open','high','low','close'}.issubset(df.columns):
        times = mdates.date2num(df.index.to_pydatetime())
        o,h,l,c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
        quotes = np.column_stack([times, o, h, l, c])
        try:
            from mplfinance.original_flavor import candlestick_ohlc
            candlestick_ohlc(
                ax_k, quotes,
                width=(df.index[1]-df.index[0]).seconds/86400*0.8,
                colorup='lime', colordown='red', alpha=0.7
            )
        except ImportError:
            ax_k.plot(df.index, c, color='white', label='Close', linewidth=1)
    else:
        # fallback
        for var in ['LFrame_ohlc5_series','close']:
            s = getattr(multFramevp_poc, var, None)
            if isinstance(s, pd.Series):
                ax_k.plot(s.index, s.values,
                          color='lightgray', linewidth=1, label=var)
                break

    # 2) 其它 series
    colors = { # … 同上略 …
        'LFrame_vwap_poc':'yellow','SFrame_vwap_poc':'purple',
        # …
    }
    for var in vars_to_plot:
        s = getattr(multFramevp_poc, var, None)
        if isinstance(s, pd.Series) and not s.isna().all():
            ax_k.plot(
                s.index, s.values,
                label=var, color=colors.get(var,'white'),
                linewidth=2 if 'HFrame' not in var else 1,
                linestyle='-' if '_poc' in var else 'dotted'
            )

    # 3) 色带
    def _fill(a,b,idx,color):
        ax_k.fill_between(
            idx, a.values, b.values,
            color=color, alpha=0.2
        )
    up1 = getattr(multFramevp_poc,'HFrame_vwap_up_getin',None)
    up2 = getattr(multFramevp_poc,'HFrame_vwap_up_sl',None)
    if isinstance(up1,pd.Series) and isinstance(up2,pd.Series):
        _fill(up1, up2, up1.index, "hotpink")
    dn1 = getattr(multFramevp_poc,'HFrame_vwap_down_sl',None)
    dn2 = getattr(multFramevp_poc,'HFrame_vwap_down_getin',None)
    if isinstance(dn1,pd.Series) and isinstance(dn2,pd.Series):
        _fill(dn1, dn2, dn1.index, "deepskyblue")

    # 4) 成交量
    if isinstance(df, pd.DataFrame) and 'vol' in df.columns:
        ax_vol.bar(df.index, df['vol'].values,
                   width=(df.index[1]-df.index[0]).seconds/86400*0.8,
                   color='dodgerblue', alpha=0.6)

    # 5) 格式化 x 轴
    ax_vol.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    # fig.autofmt_xdate(rotation=10)
    for ax in (ax_k, ax_vol):
        ax.xaxis.set_tick_params(rotation=10)

    # 6) 其余美化略……
    ax_k.set_title(f"{symbol} vp_poc/VWAP - {datetime.now()}", color='w')
    #ax_k.legend(loc='upper left', facecolor='black', labelcolor='white')
    ax_k.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,  # 可自行调节，一行排4~5个
        fontsize="small",
        facecolor="black",
        labelcolor="white"
    )
    ax_k.grid(True, alpha=0.2)
    ax_vol.grid(True, alpha=0.2)

    if save_to_file:
        os.makedirs("plots", exist_ok=True)
        fn = f"plots/{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(fn, facecolor=fig.get_facecolor())
        plt.close(fig)
        print("Saved to", fn)
    else:
        return fig


def plot_liquidation_vp(liquidation_details, bins=100, output_dir='plots_liquidation', filename='vp.png'):
        """
        Plot a volume profile of liquidation orders, separating long and short liquidations.
        
        Parameters:
        - liquidation_details: list of dicts, each with keys 'bkPx', 'sz', 'posSide'
        - bins: int or sequence, number of price bins or bin edges
        - output_dir: directory to save the plot
        - filename: name of the output image file
        """
        # Create DataFrame
        df = pd.DataFrame(liquidation_details)
        df['bkPx'] = df['bkPx'].astype(float)
        df['sz']   = df['sz'].astype(float)

        # Separate long and short liquidations
        df_long  = df[df['posSide'] == 'long']
        df_short = df[df['posSide'] == 'short']

        # Compute histograms
        hist_long, bin_edges = np.histogram(df_long['bkPx'], bins=bins, weights=df_long['sz'])
        hist_short, _         = np.histogram(df_short['bkPx'], bins=bin_edges, weights=df_short['sz'])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = bin_edges[1] - bin_edges[0]

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        import matplotlib.pyplot as plt
        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(bin_centers, hist_long,  width=width, color='red',   alpha=0.6, label='bkLoss size (long)')
        plt.bar(bin_centers, -hist_short, width=width, color='blue',  alpha=0.6, label='bkLoss size (short)')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.xlabel('bkLoss (Price)')
        plt.ylabel('bkLoss (Size)')
        plt.title('bkLoss (Liquidation Volume Profile)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path