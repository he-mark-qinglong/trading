"""
General constant string used in VN Trader.
"""

from enum import Enum


class Direction(Enum):
    """
    Direction of order/trade/position.
    """
    LONG = "buy"
    SHORT = "sell"
    NET = "net"
    UNKNOW = "unknow"
    COVERLONG = 'coverbuy'
    COVERSHORT = 'coversell'

class Offset(Enum):
    """
    Offset of order/trade.
    """
    NONE = ""
    OPEN = "open"
    CLOSE = "close"
    CLOSETODAY = "closetoday"
    CLOSEYESTERDAY = "closeyestoday"


class Status(Enum):
    """
    Order status.
    """
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    NOTTRADED = "nottrade"
    PARTTRADED = "parttraded"
    ALLTRADED = "alltrade"
    PARTCANCELED = "partcanceled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    """
    Order type.
    """
    LIMIT = "limit"
    MARKET = "market"
    STOP = "STOP"
    FAK = "FAK"
    FOK = "FOK"


class OptionType(Enum):
    """
    Option type.
    """
    CALL = "call"
    PUT = "put"


class Exchange(Enum):
    """
    Exchange.
    """
    # Chinese
    CFFEX = "CFFEX"         # China Financial Futures Exchange
    SHFE = "SHFE"           # Shanghai Futures Exchange
    CZCE = "CZCE"           # Zhengzhou Commodity Exchange
    DCE = "DCE"             # Dalian Commodity Exchange
    INE = "INE"             # Shanghai International Energy Exchange
    SSE = "SSE"             # Shanghai Stock Exchange
    SZSE = "SZSE"           # Shenzhen Stock Exchange
    SGE = "SGE"             # Shanghai Gold Exchange
    WXE = "WXE"             # Wuxi Steel Exchange

    # Global
    SMART = "SMART"         # Smart Router for US stocks
    NYMEX = "NYMEX"         # New York Mercantile Exchange
    GLOBEX = "GLOBEX"       # Globex of CME
    IDEALPRO = "IDEALPRO"   # Forex ECN of Interactive Brokers
    CME = "CME"             # Chicago Mercantile Exchange
    ICE = "ICE"             # Intercontinental Exchange
    SEHK = "SEHK"           # Stock Exchange of Hong Kong
    HKFE = "HKFE"           # Hong Kong Futures Exchange
    SGX = "SGX"             # Singapore Global Exchange
    CBOT = "CBT"            # Chicago Board of Trade
    DME = "DME"             # Dubai Mercantile Exchange
    EUREX = "EUX"           # Eurex Exchange
    APEX = "APEX"           # Asia Pacific Exchange
    LME = "LME"             # London Metal Exchange
    BMD = "BMD"             # Bursa Malaysia Derivatives
    TOCOM = "TOCOM"         # Tokyo Commodity Exchange
    EUNX = "EUNX"           # Euronext Exchange

    # CryptoCurrency
    BITMEX = "BITMEX"
    OKEX = "OKEX"
    HUOBI = "HUOBI"
    BITFINEX = "BITFINEX"
    BITKER = "BITKER"
    BITKERC = "BITKERC"
    BINANCE = "BINANCE"
    ETHEX = "ETHEX"
    NEWTON = "NEWTON"
    MXC = "MXC"
    BITMAX = "BITMAX"
    WTZ = "WTZ"
    BETAEX = 'BETAEX'
    GOOD = "GOOD"
    DERIBIT = 'DERIBIT'
    FMEX='FMEX'
    COINW = 'coinw'
    BIGONE = 'bigone'
    VC = 'VC'
    CRYPTO = 'CRYPTO'


class Interval(Enum):
    """
    Interval of bar data.
    """
    MINUTE = "1m"
    HOUR = "1h"
    DAILY = "d"
    WEEKLY = "w"