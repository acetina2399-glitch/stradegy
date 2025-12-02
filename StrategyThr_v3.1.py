# StrategyThr v3.1 - 4状态盘中交易策略
# 适用平台: PTrade
# 标的: 603305.SS (旭升集团)
# 版本更新:
#   v3.1 - 盘中交易增强版
#   1. 移除原T+0逻辑，改用状态感知的盘中交易
#   2. STRONG_LONG: 回落加仓，反弹/死叉减仓
#   3. WEAK_LONG: 高抛低吸，死叉停止买入
#   4. WEAK_SHORT: 逢高减仓，回落补仓，死叉停止买入
#   5. STRONG_SHORT: 清仓，不参与盘中交易
#   6. 价格档位机制: 每档只触发一次
#
#   v3.1.1 - 盘中实时监控增强 (方案C)
#   7. 分时MACD实时计算: 盘中死叉/金叉实时检测
#   8. 量价异常监控: 放量/缩量/急跌预警
#   9. 极端行情保护: 涨跌停/大幅波动/开盘跳空保护
#
#   v3.1.2 - 技术形态系数矩阵
#   10. 6类技术信号系数: MACD+量能/RSI/背离/KDJ/BIAS/量价偏离
#   11. 执行仓位 = 基础仓位 × 综合系数
#   12. 信号衰减机制: 信号强度随时间递减

import numpy as np
from datetime import datetime, timedelta

# ==================== 全局参数配置 ====================
class Config:
    """策略参数配置"""

    # 基础标的设置
    STOCK = '603305.SS'
    BENCHMARK = '000300.SS'

    # ===== 数据参数 =====
    TREND_LOOKBACK = 250
    VOLUME_MA_PERIOD = 20
    PRICE_MOMENTUM_PERIOD = 5

    # ===== MACD参数 =====
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9

    # ===== RSI参数 =====
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 80
    RSI_OVERSOLD = 30
    RSI_STRONG = 50

    # ===== 道氏理论参数 =====
    DOW_SWING_WINDOW = 5

    # ===== 背离检测参数 =====
    DIVERGENCE_LOOKBACK = 20

    # ===== 仓位控制参数 =====
    POSITION_CHANGE_THRESHOLD = 0.08
    POSITION_SMOOTH_MAX_CHANGE = 0.20

    # ===== 4状态仓位配置 =====
    POSITION_STRONG_LONG = 0.90
    POSITION_WEAK_LONG = 0.60
    POSITION_WEAK_SHORT = 0.30
    POSITION_STRONG_SHORT = 0.0

    # ===== 评分阈值 =====
    VOL_RATIO_HIGH = 1.2
    VOL_RATIO_LOW = 0.8
    MOM_STRONG_UP = 0.02
    MOM_STRONG_DOWN = -0.02
    MACD_EXPAND_RATIO = 1.5

    # ===== 盘中交易参数 =====
    # STRONG_LONG
    SL_BUY_DROP_FIRST = 0.01       # 首次买入: 回落1%
    SL_BUY_DROP_STEP = 0.01        # 后续买入: 每档回落1%
    SL_SELL_RISE_FIRST = 0.02     # 首次卖出: 反弹2%
    SL_SELL_RISE_STEP = 0.01      # 后续卖出: 每档上涨1%
    SL_MAX_POSITION = 2.0          # 最大仓位200%

    # WEAK_LONG
    WL_SELL_RISE_FIRST = 0.02     # 首次卖出: 上涨2%
    WL_SELL_RISE_STEP = 0.01      # 后续卖出: 每档上涨1%
    WL_BUY_DROP_FIRST = 0.01      # 首次买入: 回落1%
    WL_BUY_DROP_STEP = 0.01       # 后续买入: 每档回落1%

    # WEAK_SHORT
    WS_SELL_RISE_FIRST = 0.02     # 首次卖出: 上涨2%
    WS_SELL_RISE_STEP = 0.01      # 后续卖出: 每档上涨1%
    WS_FIRST_SELL_RATIO = 0.20    # 首次卖出20%
    WS_SELL_RATIO = 0.10          # 后续卖出10%
    WS_BUY_DROP_FIRST = 0.01      # 首次买入: 回落1%
    WS_BUY_DROP_STEP = 0.01       # 后续买入: 每档回落1%

    # 通用
    TRADE_RATIO = 0.10             # 每次交易10%总资产

    # ===== 风控参数 (v3.1.4 简化版) =====
    # 盘中极端保护: 日内高点回落超过此比例则清仓
    INTRADAY_DROP_THRESHOLD = 0.12
    # 账户止损: 连续亏损天数 + 累计亏损比例
    ACCOUNT_LOSS_DAYS = 3            # 连续亏损天数
    ACCOUNT_LOSS_THRESHOLD = 0.08    # 累计亏损阈值
    # 统一冷却期
    COOLDOWN_DAYS = 5

    # ===== 交易时间 =====
    TRADE_START_HOUR = 9
    TRADE_START_MINUTE = 35
    TRADE_END_HOUR = 14
    TRADE_END_MINUTE = 50

    # ===== 盘中MACD参数 (分钟级) =====
    INTRADAY_MACD_FAST = 12
    INTRADAY_MACD_SLOW = 26
    INTRADAY_MACD_SIGNAL = 9
    INTRADAY_MACD_MIN_BARS = 30  # 至少30根分钟K线才计算

    # ===== 量价异常阈值 =====
    REALTIME_VOL_RATIO_HIGH = 3.0    # 分钟放量阈值
    CUMULATIVE_VOL_RATIO_HIGH = 2.0  # 累计放量阈值
    CUMULATIVE_VOL_RATIO_LOW = 0.5   # 累计缩量阈值
    MOMENTUM_ALERT_THRESHOLD = 0.05  # 5%动量预警
    PAUSE_AFTER_PLUNGE_MINUTES = 30  # 急跌后暂停买入时间

    # ===== 极端行情阈值 =====
    LIMIT_UP_THRESHOLD = 0.098       # 涨停阈值 (9.8%)
    LIMIT_DOWN_THRESHOLD = -0.098    # 跌停阈值 (-9.8%)
    VOLATILITY_PAUSE_THRESHOLD = 0.03  # 5分钟内3%波动触发暂停
    VOLATILITY_PAUSE_MINUTES = 5     # 波动暂停时间
    OPEN_GAP_THRESHOLD = 0.03        # 开盘跳空3%阈值
    DELAYED_START_MINUTE = 45        # 跳空后延迟至9:45开始

    # ===== KDJ参数 =====
    KDJ_N = 9                        # RSV周期
    KDJ_M1 = 3                       # K平滑周期
    KDJ_M2 = 3                       # D平滑周期
    KDJ_OVERBOUGHT = 80              # 超买阈值
    KDJ_OVERSOLD = 20                # 超卖阈值
    KDJ_HIGH_CROSS = 70              # 高位死叉阈值
    KDJ_LOW_CROSS = 30               # 低位金叉阈值

    # ===== BIAS参数 =====
    BIAS_PERIOD = 20                 # BIAS计算周期
    BIAS_SEVERE_OVERBOUGHT = 0.08    # 严重超买 >8%
    BIAS_OVERBOUGHT = 0.05           # 超买 5%~8%
    BIAS_OVERSOLD = -0.03            # 超卖 -3%~-5%
    BIAS_SEVERE_OVERSOLD = -0.05     # 严重超卖 <-5%

    # ===== 量价偏离参数 =====
    VP_DIVERGE_DAYS = 3              # 量价偏离检测天数
    VP_STAGNATION_VOL = 2.0          # 放量滞涨量比阈值
    VP_STAGNATION_GAIN = 0.01        # 放量滞涨涨幅阈值
    VP_SHRINK_VOL = 0.6              # 缩量阴跌量比阈值

    # ===== 信号系数配置 =====
    # MACD+量能系数
    COEF_MACD_VOL_GOLDEN = 1.25      # 放量金叉
    COEF_MACD_GOLDEN = 1.10          # 普通金叉
    COEF_MACD_GOLDEN_WEAK = 1.00     # 缩量金叉(无效)
    COEF_MACD_DEAD = 0.70            # 普通死叉
    COEF_MACD_VOL_DEAD = 0.50        # 放量死叉
    COEF_MACD_DEAD_BELOW_ZERO = 0.30 # 零轴下死叉

    # RSI系数
    COEF_RSI_OVERSOLD_REBOUND = 1.15 # 超卖反弹
    COEF_RSI_OVERBOUGHT_DROP = 0.60  # 超买回落
    COEF_RSI_TOP_DULL = 0.50         # 顶部钝化
    COEF_RSI_BOTTOM_DULL = 1.20      # 底部钝化

    # 背离系数
    COEF_BOTTOM_DIVERGE = 1.25       # 底背离
    COEF_TOP_DIVERGE = 0.40          # 顶背离

    # KDJ系数
    COEF_KDJ_LOW_GOLDEN = 1.15       # 低位金叉
    COEF_KDJ_HIGH_DEAD = 0.60        # 高位死叉
    COEF_KDJ_OVERBOUGHT = 0.80       # 超买区
    COEF_KDJ_OVERSOLD = 1.10         # 超卖区

    # BIAS系数
    COEF_BIAS_SEVERE_OB = 0.50       # 严重超买
    COEF_BIAS_OB = 0.75              # 超买
    COEF_BIAS_OS = 1.10              # 超卖
    COEF_BIAS_SEVERE_OS = 1.20       # 严重超卖

    # 量价偏离系数
    COEF_VP_TOP_DIVERGE = 0.50       # 量价背离(顶)
    COEF_VP_BOTTOM_DIVERGE = 1.15    # 量价背离(底)
    COEF_VP_STAGNATION = 0.40        # 放量滞涨
    COEF_VP_SHRINK_DROP = 0.80       # 缩量阴跌


# ==================== 工具函数 ====================
class Utils:
    """通用工具函数"""

    @staticmethod
    def calculate_volatility(prices, period=20):
        if len(prices) < period + 1:
            return 0.5
        recent_prices = prices[-(period + 1):]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        raw_vol = np.std(returns) * np.sqrt(252)
        normalized = (raw_vol - 0.1) / 0.7
        return max(0, min(1, normalized))

    @staticmethod
    def safe_divide(a, b, default=0):
        return a / b if b != 0 else default

    @staticmethod
    def validate_prices(prices, min_length=20):
        if prices is None or len(prices) < min_length:
            return False, None, "数据不足"
        prices_arr = np.array(prices, dtype=float)
        nan_count = np.sum(np.isnan(prices_arr))
        if nan_count / len(prices_arr) > 0.1:
            return False, None, "NaN过多"
        for i in range(len(prices_arr)):
            if np.isnan(prices_arr[i]) or prices_arr[i] <= 0:
                if i > 0:
                    prices_arr[i] = prices_arr[i-1]
        return True, prices_arr, None

    @staticmethod
    def validate_volumes(volumes, min_length=20):
        if volumes is None or len(volumes) < min_length:
            return False, None, "数据不足"
        volumes_arr = np.array(volumes, dtype=float)
        volumes_arr = np.nan_to_num(volumes_arr, nan=0.0)
        return True, volumes_arr, None


# ==================== 道氏理论分析器 ====================
class DowTheoryAnalyzer:
    """道氏理论趋势确认"""

    @staticmethod
    def find_swing_points(prices, window=5):
        highs = []
        lows = []
        for i in range(window, len(prices) - window):
            is_high = True
            is_low = True
            for j in range(1, window + 1):
                if prices[i] <= prices[i - j] or prices[i] <= prices[i + j]:
                    is_high = False
                if prices[i] >= prices[i - j] or prices[i] >= prices[i + j]:
                    is_low = False
            if is_high:
                highs.append((i, prices[i]))
            if is_low:
                lows.append((i, prices[i]))
        return highs, lows

    @staticmethod
    def analyze(prices, window=None):
        if window is None:
            window = Config.DOW_SWING_WINDOW
        if len(prices) < window * 4:
            return False, 'NONE', {}
        highs, lows = DowTheoryAnalyzer.find_swing_points(prices, window)
        if len(highs) < 2 or len(lows) < 2:
            return False, 'NONE', {'reason': '摆动点不足'}
        recent_highs = highs[-2:]
        recent_lows = lows[-2:]
        higher_highs = recent_highs[1][1] > recent_highs[0][1]
        higher_lows = recent_lows[1][1] > recent_lows[0][1]
        lower_highs = recent_highs[1][1] < recent_highs[0][1]
        lower_lows = recent_lows[1][1] < recent_lows[0][1]
        details = {
            'highs': recent_highs,
            'lows': recent_lows,
            'higher_highs': higher_highs,
            'higher_lows': higher_lows,
        }
        if higher_highs and higher_lows:
            return True, 'UP', details
        elif lower_highs and lower_lows:
            return True, 'DOWN', details
        else:
            return False, 'NONE', details


# ==================== MACD计算器 ====================
class MACDCalculator:
    """MACD指标计算"""

    @staticmethod
    def _ema(prices, period):
        n = len(prices)
        if n == 0:
            return np.array([])
        result = np.empty(n)
        result[0] = prices[0]
        k = 2.0 / (period + 1)
        k1 = 1.0 - k
        for i in range(1, n):
            result[i] = prices[i] * k + result[i-1] * k1
        return result

    @staticmethod
    def calculate(prices):
        min_length = Config.MACD_SLOW + Config.MACD_SIGNAL
        if len(prices) < min_length:
            return None, None, None
        ema_fast = MACDCalculator._ema(prices, Config.MACD_FAST)
        ema_slow = MACDCalculator._ema(prices, Config.MACD_SLOW)
        dif = ema_fast - ema_slow
        n = len(dif)
        dea = np.empty(n)
        dea[0] = dif[0]
        k = 2.0 / (Config.MACD_SIGNAL + 1)
        k1 = 1.0 - k
        for i in range(1, n):
            dea[i] = dif[i] * k + dea[i-1] * k1
        histogram = 2.0 * (dif - dea)
        return dif, dea, histogram

    @staticmethod
    def detect_cross(dif, dea):
        """检测MACD金叉死叉"""
        if dif is None or dea is None or len(dif) < 2:
            return None
        curr_above = dif[-1] > dea[-1]
        prev_above = dif[-2] > dea[-2]
        if curr_above and not prev_above:
            return 'golden'
        elif not curr_above and prev_above:
            return 'dead'
        return None

    @staticmethod
    def check_dead_cross_state(dif, dea):
        """检查当前是否处于死叉状态 (DIF < DEA)"""
        if dif is None or dea is None or len(dif) < 1:
            return False
        return dif[-1] < dea[-1]


# ==================== RSI计算器 ====================
class RSICalculator:
    """RSI指标计算"""

    @staticmethod
    def calculate(prices, period=None):
        if period is None:
            period = Config.RSI_PERIOD
        if len(prices) < period + 1:
            return 50
        deltas = np.diff(prices[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


# ==================== 背离检测器 ====================
class DivergenceDetector:
    """价格与MACD背离检测"""

    @staticmethod
    def detect(prices, macd_hist, lookback=None):
        if lookback is None:
            lookback = Config.DIVERGENCE_LOOKBACK
        if len(prices) < lookback or macd_hist is None or len(macd_hist) < lookback:
            return False, False
        recent_prices = prices[-lookback:]
        recent_hist = macd_hist[-lookback:]
        price_high_idx = np.argmax(recent_prices)
        hist_high_idx = np.argmax(recent_hist)
        top_diverge = (price_high_idx > lookback * 0.6 and
                      hist_high_idx < lookback * 0.5 and
                      recent_prices[-1] > recent_prices[0] and
                      recent_hist[-1] < recent_hist[hist_high_idx] * 0.8)
        price_low_idx = np.argmin(recent_prices)
        hist_low_idx = np.argmin(recent_hist)
        bottom_diverge = (price_low_idx > lookback * 0.6 and
                         hist_low_idx < lookback * 0.5 and
                         recent_prices[-1] < recent_prices[0] and
                         recent_hist[-1] > recent_hist[hist_low_idx] * 0.8)
        return top_diverge, bottom_diverge


# ==================== KDJ计算器 ====================
class KDJCalculator:
    """KDJ随机指标计算"""

    @staticmethod
    def calculate(prices, highs=None, lows=None):
        """
        计算KDJ指标
        参数:
            prices: 收盘价序列
            highs: 最高价序列 (可选，默认用收盘价)
            lows: 最低价序列 (可选，默认用收盘价)
        返回:
            (K, D, J, K序列, D序列) 或 (50, 50, 50, None, None) 如果数据不足
        """
        n = Config.KDJ_N
        m1 = Config.KDJ_M1
        m2 = Config.KDJ_M2

        if len(prices) < n + m1 + m2:
            return 50, 50, 50, None, None

        # 如果没有提供高低价，用收盘价估算
        if highs is None:
            highs = prices
        if lows is None:
            lows = prices

        # 计算RSV
        rsv_list = []
        for i in range(n - 1, len(prices)):
            high_n = np.max(highs[i - n + 1:i + 1])
            low_n = np.min(lows[i - n + 1:i + 1])
            if high_n == low_n:
                rsv = 50
            else:
                rsv = (prices[i] - low_n) / (high_n - low_n) * 100
            rsv_list.append(rsv)

        # 计算K值 (RSV的M1日EMA)
        k_list = [rsv_list[0]]
        for i in range(1, len(rsv_list)):
            k = (rsv_list[i] + (m1 - 1) * k_list[-1]) / m1
            k_list.append(k)

        # 计算D值 (K的M2日EMA)
        d_list = [k_list[0]]
        for i in range(1, len(k_list)):
            d = (k_list[i] + (m2 - 1) * d_list[-1]) / m2
            d_list.append(d)

        # 计算J值
        j_list = [3 * k_list[i] - 2 * d_list[i] for i in range(len(k_list))]

        k = k_list[-1]
        d = d_list[-1]
        j = j_list[-1]

        return k, d, j, np.array(k_list), np.array(d_list)

    @staticmethod
    def detect_cross(k_series, d_series):
        """
        检测KDJ金叉死叉
        返回: ('golden', K值) / ('dead', K值) / (None, K值)
        """
        if k_series is None or d_series is None or len(k_series) < 2:
            return None, 50

        k_curr, k_prev = k_series[-1], k_series[-2]
        d_curr, d_prev = d_series[-1], d_series[-2]

        # 金叉: K从下向上穿过D
        if k_prev <= d_prev and k_curr > d_curr:
            return 'golden', k_curr
        # 死叉: K从上向下穿过D
        elif k_prev >= d_prev and k_curr < d_curr:
            return 'dead', k_curr

        return None, k_curr


# ==================== BIAS计算器 ====================
class BIASCalculator:
    """乖离率计算"""

    @staticmethod
    def calculate(prices, period=None):
        """
        计算BIAS乖离率
        BIAS = (当前价 - N日均线) / N日均线 × 100%
        返回: bias值 (小数形式，如0.05表示5%)
        """
        if period is None:
            period = Config.BIAS_PERIOD

        if len(prices) < period:
            return 0

        ma = np.mean(prices[-period:])
        if ma == 0:
            return 0

        bias = (prices[-1] - ma) / ma
        return bias

    @staticmethod
    def get_state(bias):
        """
        获取BIAS状态
        返回: 'severe_overbought' / 'overbought' / 'oversold' / 'severe_oversold' / 'normal'
        """
        if bias >= Config.BIAS_SEVERE_OVERBOUGHT:
            return 'severe_overbought'
        elif bias >= Config.BIAS_OVERBOUGHT:
            return 'overbought'
        elif bias <= Config.BIAS_SEVERE_OVERSOLD:
            return 'severe_oversold'
        elif bias <= Config.BIAS_OVERSOLD:
            return 'oversold'
        else:
            return 'normal'


# ==================== 量价偏离度分析器 ====================
class VolumePriceDivergence:
    """量价偏离度分析"""

    @staticmethod
    def analyze(prices, volumes, vol_ma_period=20):
        """
        分析量价偏离情况
        返回: (状态, 详情字典)
        状态: 'top_diverge' / 'bottom_diverge' / 'stagnation' / 'shrink_drop' / 'normal'
        """
        days = Config.VP_DIVERGE_DAYS
        if len(prices) < days + vol_ma_period or len(volumes) < days + vol_ma_period:
            return 'normal', {}

        # 计算量比
        vol_ma = np.mean(volumes[-(vol_ma_period + days):-days]) if days > 0 else np.mean(volumes[-vol_ma_period:])
        recent_vol_avg = np.mean(volumes[-days:])
        vol_ratio = recent_vol_avg / vol_ma if vol_ma > 0 else 1.0

        # 计算近期涨跌
        price_change = (prices[-1] - prices[-days - 1]) / prices[-days - 1]

        # 计算每日涨跌
        daily_changes = []
        for i in range(-days, 0):
            if i == -days:
                change = (prices[i] - prices[i - 1]) / prices[i - 1]
            else:
                change = (prices[i] - prices[i - 1]) / prices[i - 1]
            daily_changes.append(change)

        # 连跌天数
        down_days = sum(1 for c in daily_changes if c < 0)

        details = {
            'vol_ratio': vol_ratio,
            'price_change': price_change,
            'down_days': down_days,
        }

        # 1. 放量滞涨: 量比>2 且 涨幅<1%
        if vol_ratio > Config.VP_STAGNATION_VOL and price_change < Config.VP_STAGNATION_GAIN:
            return 'stagnation', details

        # 2. 量价背离(顶): 价涨 + 量缩 + 连续
        if price_change > 0.02 and vol_ratio < 0.8 and down_days == 0:
            # 价格上涨但量能萎缩
            return 'top_diverge', details

        # 3. 量价背离(底): 价跌 + 量缩 + 企稳迹象
        if price_change < -0.02 and vol_ratio < 0.8:
            # 检查是否企稳 (最后一天跌幅收窄或翻红)
            if len(daily_changes) > 0 and daily_changes[-1] > daily_changes[0]:
                return 'bottom_diverge', details

        # 4. 缩量阴跌: 量比<0.6 且 连跌3天
        if vol_ratio < Config.VP_SHRINK_VOL and down_days >= days:
            return 'shrink_drop', details

        return 'normal', details


# ==================== 信号得分计算器 (简化版) ====================
class SignalScorer:
    """
    信号得分计算器
    将6类技术信号转换为统一得分: 看多(+1) / 中性(0) / 看空(-1)
    总分范围: -6 ~ +6
    """

    @staticmethod
    def get_macd_score(macd_cross, vol_ratio, dif, dea):
        """
        MACD+量能得分
        返回: (得分, 信号名称, 是否清仓信号)
        """
        if macd_cross == 'golden':
            if vol_ratio > 1.5:
                return +1, 'MACD放量金叉', False
            else:
                return +1, 'MACD金叉', False

        elif macd_cross == 'dead':
            # 零轴下死叉 = 清仓信号
            if dif is not None and dea is not None:
                if dif[-1] < 0 and dea[-1] < 0:
                    return -1, 'MACD零轴下死叉', True
            if vol_ratio > 1.5:
                return -1, 'MACD放量死叉', False
            else:
                return -1, 'MACD死叉', False

        return 0, None, False

    @staticmethod
    def get_rsi_score(rsi, rsi_history=None):
        """
        RSI得分
        返回: (得分, 信号名称, 是否清仓信号)
        """
        if rsi_history is not None and len(rsi_history) >= 2:
            rsi_prev = rsi_history[-2]
            # 超卖反弹
            if rsi_prev < 20 and rsi >= 30:
                return +1, 'RSI超卖反弹', False
            # 超买回落
            if rsi_prev > 80 and rsi <= 70:
                return -1, 'RSI超买回落', False

            # 顶部钝化 (连续3天RSI>80)
            if len(rsi_history) >= 3:
                if all(r > 80 for r in rsi_history[-3:]):
                    return -1, 'RSI顶部钝化', False
                # 底部钝化
                if all(r < 20 for r in rsi_history[-3:]):
                    return +1, 'RSI底部钝化', False

        return 0, None, False

    @staticmethod
    def get_diverge_score(top_diverge, bottom_diverge):
        """
        背离得分
        返回: (得分, 信号名称, 是否清仓信号)
        """
        if top_diverge:
            return -1, '顶背离', True  # 顶背离 = 清仓信号
        if bottom_diverge:
            return +1, '底背离', False
        return 0, None, False

    @staticmethod
    def get_kdj_score(k, d, kdj_cross, k_value):
        """
        KDJ得分
        返回: (得分, 信号名称, 是否清仓信号)
        """
        if kdj_cross == 'golden' and k_value < Config.KDJ_LOW_CROSS:
            return +1, 'KDJ低位金叉', False
        if kdj_cross == 'dead' and k_value > Config.KDJ_HIGH_CROSS:
            return -1, 'KDJ高位死叉', False

        # 超买超卖区域
        if k > Config.KDJ_OVERBOUGHT and d > Config.KDJ_OVERBOUGHT:
            return -1, 'KDJ超买区', False
        if k < Config.KDJ_OVERSOLD and d < Config.KDJ_OVERSOLD:
            return +1, 'KDJ超卖区', False

        return 0, None, False

    @staticmethod
    def get_bias_score(bias):
        """
        BIAS得分
        返回: (得分, 信号名称, 是否清仓信号)
        """
        if bias >= Config.BIAS_SEVERE_OVERBOUGHT:
            return -1, 'BIAS严重超买', False
        elif bias >= Config.BIAS_OVERBOUGHT:
            return -1, 'BIAS超买', False
        elif bias <= Config.BIAS_SEVERE_OVERSOLD:
            return +1, 'BIAS严重超卖', False
        elif bias <= Config.BIAS_OVERSOLD:
            return +1, 'BIAS超卖', False
        return 0, None, False

    @staticmethod
    def get_vp_score(vp_state):
        """
        量价偏离得分
        返回: (得分, 信号名称, 是否清仓信号)
        """
        if vp_state == 'stagnation':
            return -1, '放量滞涨', True  # 放量滞涨 = 清仓信号
        elif vp_state == 'top_diverge':
            return -1, '量价顶背离', False
        elif vp_state == 'bottom_diverge':
            return +1, '量价底背离', False
        elif vp_state == 'shrink_drop':
            return -1, '缩量阴跌', False
        return 0, None, False

    @staticmethod
    def calculate_total_score(macd_cross, vol_ratio, dif, dea,
                               rsi, rsi_history,
                               top_diverge, bottom_diverge,
                               k, d, kdj_cross, k_value,
                               bias, vp_state):
        """
        计算总得分
        返回: (总分, 触发信号列表, 是否有清仓信号)
        """
        total_score = 0
        signals = []
        has_liquidation_signal = False

        # 1. MACD+量能
        score, sig, is_liq = SignalScorer.get_macd_score(macd_cross, vol_ratio, dif, dea)
        total_score += score
        if sig:
            signals.append(sig)
        if is_liq:
            has_liquidation_signal = True

        # 2. RSI
        score, sig, is_liq = SignalScorer.get_rsi_score(rsi, rsi_history)
        total_score += score
        if sig:
            signals.append(sig)
        if is_liq:
            has_liquidation_signal = True

        # 3. 背离
        score, sig, is_liq = SignalScorer.get_diverge_score(top_diverge, bottom_diverge)
        total_score += score
        if sig:
            signals.append(sig)
        if is_liq:
            has_liquidation_signal = True

        # 4. KDJ
        score, sig, is_liq = SignalScorer.get_kdj_score(k, d, kdj_cross, k_value)
        total_score += score
        if sig:
            signals.append(sig)
        if is_liq:
            has_liquidation_signal = True

        # 5. BIAS
        score, sig, is_liq = SignalScorer.get_bias_score(bias)
        total_score += score
        if sig:
            signals.append(sig)
        if is_liq:
            has_liquidation_signal = True

        # 6. 量价偏离
        score, sig, is_liq = SignalScorer.get_vp_score(vp_state)
        total_score += score
        if sig:
            signals.append(sig)
        if is_liq:
            has_liquidation_signal = True

        return total_score, signals, has_liquidation_signal


# ==================== 信号状态机 ====================
class SignalStateMachine:
    """
    信号状态机
    管理信号状态转换和系数计算

    状态:
      NORMAL      - 正常交易
      ALERT       - 警戒状态
      LIQUIDATION - 清仓状态
      RECOVERY    - 恢复状态

    状态转换规则:
      NORMAL + (得分≤-3 或 清仓信号) → LIQUIDATION
      NORMAL + 得分-1~-2 → ALERT
      ALERT + 得分≤-1 → LIQUIDATION
      ALERT + 得分≥1 → NORMAL
      LIQUIDATION + 连续2天得分≥3 → RECOVERY
      RECOVERY + 得分≥1 → NORMAL
      RECOVERY + 得分≤-1 → LIQUIDATION
    """

    # 状态常量
    NORMAL = "NORMAL"
    ALERT = "ALERT"
    LIQUIDATION = "LIQUIDATION"
    RECOVERY = "RECOVERY"

    # 系数配置
    COEF_STRONG_BULL = 1.50    # 强烈看多
    COEF_MILD_BULL = 1.10      # 温和看多
    COEF_ALERT = 0.50          # 警戒状态
    COEF_ALERT_RECOVER = 0.80  # 警戒恢复
    COEF_RECOVERY = 0.30       # 恢复状态
    COEF_RECOVERY_OK = 0.50    # 恢复确认
    COEF_LIQUIDATION = 0.0     # 清仓

    def __init__(self):
        self.state = self.NORMAL
        self.consecutive_strong_bull_days = 0  # 连续强看多天数
        self.days_in_liquidation = 0           # 清仓状态天数

    def reset(self):
        """重置状态机"""
        self.state = self.NORMAL
        self.consecutive_strong_bull_days = 0
        self.days_in_liquidation = 0

    def get_state_name_cn(self):
        """获取状态中文名"""
        names = {
            self.NORMAL: "正常",
            self.ALERT: "警戒",
            self.LIQUIDATION: "清仓",
            self.RECOVERY: "恢复",
        }
        return names.get(self.state, "未知")

    def process(self, total_score, has_liquidation_signal):
        """
        处理信号得分，返回系数和新状态

        参数:
            total_score: 信号总得分 (-6 ~ +6)
            has_liquidation_signal: 是否有清仓信号

        返回:
            (系数, 状态变化描述)
        """
        old_state = self.state
        coef = 1.0
        transition_desc = None

        # ===== NORMAL 状态 =====
        if self.state == self.NORMAL:
            if has_liquidation_signal or total_score <= -3:
                # 直接清仓
                self.state = self.LIQUIDATION
                self.days_in_liquidation = 1
                self.consecutive_strong_bull_days = 0
                coef = self.COEF_LIQUIDATION
                transition_desc = "触发清仓信号" if has_liquidation_signal else "强烈看空"

            elif total_score <= -1:
                # 进入警戒
                self.state = self.ALERT
                coef = self.COEF_ALERT
                transition_desc = "进入警戒"

            elif total_score >= 3:
                # 强烈看多
                coef = self.COEF_STRONG_BULL
                self.consecutive_strong_bull_days += 1

            elif total_score >= 1:
                # 温和看多
                coef = self.COEF_MILD_BULL
                self.consecutive_strong_bull_days = 0

            else:
                # 中性 - 不调整 (得分=0时不交易)
                coef = 1.0
                self.consecutive_strong_bull_days = 0

        # ===== ALERT 状态 =====
        elif self.state == self.ALERT:
            if total_score <= -1:
                # 恶化 → 清仓
                self.state = self.LIQUIDATION
                self.days_in_liquidation = 1
                coef = self.COEF_LIQUIDATION
                transition_desc = "警戒恶化→清仓"

            elif total_score >= 1:
                # 好转 → 正常
                self.state = self.NORMAL
                coef = self.COEF_ALERT_RECOVER
                transition_desc = "警戒恢复→正常"

            else:
                # 维持警戒 (得分=0)
                coef = self.COEF_ALERT

        # ===== LIQUIDATION 状态 =====
        elif self.state == self.LIQUIDATION:
            self.days_in_liquidation += 1

            if total_score >= 3:
                self.consecutive_strong_bull_days += 1
                # 连续2天强看多 → 恢复
                if self.consecutive_strong_bull_days >= 2:
                    self.state = self.RECOVERY
                    coef = self.COEF_RECOVERY
                    transition_desc = "开始恢复"
                else:
                    coef = self.COEF_LIQUIDATION
            else:
                # 继续清仓
                self.consecutive_strong_bull_days = 0
                coef = self.COEF_LIQUIDATION

        # ===== RECOVERY 状态 =====
        elif self.state == self.RECOVERY:
            if total_score <= -1:
                # 恢复失败 → 清仓
                self.state = self.LIQUIDATION
                self.days_in_liquidation = 1
                self.consecutive_strong_bull_days = 0
                coef = self.COEF_LIQUIDATION
                transition_desc = "恢复失败→清仓"

            elif total_score >= 1:
                # 恢复成功 → 正常
                self.state = self.NORMAL
                self.consecutive_strong_bull_days = 0
                coef = self.COEF_RECOVERY_OK
                transition_desc = "恢复成功→正常"

            else:
                # 观望 (得分=0)
                coef = self.COEF_RECOVERY

        # 构建状态变化描述
        if transition_desc is None and old_state != self.state:
            transition_desc = "%s→%s" % (old_state, self.state)

        return coef, transition_desc

    def get_status_summary(self):
        """获取状态摘要"""
        summary = {
            'state': self.state,
            'state_cn': self.get_state_name_cn(),
            'consecutive_bull_days': self.consecutive_strong_bull_days,
            'days_in_liquidation': self.days_in_liquidation if self.state == self.LIQUIDATION else 0,
        }
        return summary

    def force_liquidation(self):
        """
        强制进入清仓状态 (由风控触发)
        用于日内极端保护或账户止损触发时
        """
        self.state = self.LIQUIDATION
        self.days_in_liquidation = 1
        self.consecutive_strong_bull_days = 0
        log.warning("[状态机] 风控强制清仓，进入LIQUIDATION状态")


# ==================== 盘中实时监控器 ====================
class IntradayMonitor:
    """
    盘中实时监控器 - 方案C核心组件
    功能:
    1. 分时MACD实时计算与死叉/金叉检测
    2. 量价异常监控 (放量/缩量/急跌)
    3. 极端行情保护 (涨跌停/大幅波动/开盘跳空)
    """

    def __init__(self):
        self.reset_daily()

    def reset_daily(self):
        """每日重置"""
        # 分钟数据序列
        self.minute_prices = []       # 分钟收盘价
        self.minute_volumes = []      # 分钟成交量
        self.minute_times = []        # 分钟时间戳

        # 分时MACD
        self.intraday_dif = None
        self.intraday_dea = None
        self.intraday_macd_cross = None  # 'golden' / 'dead' / None

        # 保护状态
        self.dead_cross_active = False   # 分时死叉激活
        self.can_buy = True              # 允许买入
        self.can_sell = True             # 允许卖出
        self.pause_buy_until = None      # 暂停买入至某时间
        self.pause_all_until = None      # 暂停所有交易至某时间

        # 开盘状态
        self.open_price = 0
        self.prev_close = 0
        self.open_gap = 0                # 开盘跳空幅度
        self.delayed_start = False       # 是否延迟开始

        # 量价监控
        self.cumulative_volume = 0       # 累计成交量
        self.prev_day_volume = 0         # 昨日全天成交量
        self.last_alert_time = None      # 上次预警时间
        self.plunge_detected = False     # 急跌检测

        # 波动监控
        self.recent_prices = []          # 最近N分钟价格 (波动检测)
        self.volatility_pause_count = 0  # 波动暂停次数

    def set_prev_close(self, prev_close):
        """设置昨收价"""
        self.prev_close = prev_close

    def set_prev_day_volume(self, volume):
        """设置昨日成交量"""
        self.prev_day_volume = volume

    def check_open_gap(self, open_price):
        """检查开盘跳空"""
        self.open_price = open_price
        if self.prev_close > 0:
            self.open_gap = (open_price - self.prev_close) / self.prev_close
            if abs(self.open_gap) > Config.OPEN_GAP_THRESHOLD:
                self.delayed_start = True
                return True, self.open_gap
        return False, 0

    def update_minute_data(self, price, volume, current_time):
        """更新分钟数据"""
        self.minute_prices.append(price)
        self.minute_volumes.append(volume)
        self.minute_times.append(current_time)
        self.cumulative_volume += volume

        # 更新最近价格 (保留最近5分钟)
        self.recent_prices.append(price)
        if len(self.recent_prices) > 5:
            self.recent_prices.pop(0)

    def calculate_intraday_macd(self):
        """计算分时MACD"""
        prices = self.minute_prices
        if len(prices) < Config.INTRADAY_MACD_MIN_BARS:
            return None, None, None

        # EMA计算
        def ema(data, period):
            result = [data[0]]
            k = 2.0 / (period + 1)
            for i in range(1, len(data)):
                result.append(data[i] * k + result[-1] * (1 - k))
            return np.array(result)

        prices_arr = np.array(prices)
        ema_fast = ema(prices_arr, Config.INTRADAY_MACD_FAST)
        ema_slow = ema(prices_arr, Config.INTRADAY_MACD_SLOW)
        dif = ema_fast - ema_slow
        dea = ema(dif, Config.INTRADAY_MACD_SIGNAL)

        self.intraday_dif = dif
        self.intraday_dea = dea

        return dif, dea, 2 * (dif - dea)

    def check_intraday_macd_cross(self):
        """检测分时MACD金叉死叉"""
        if self.intraday_dif is None or len(self.intraday_dif) < 2:
            return None

        dif = self.intraday_dif
        dea = self.intraday_dea

        curr_above = dif[-1] > dea[-1]
        prev_above = dif[-2] > dea[-2]

        if curr_above and not prev_above:
            self.intraday_macd_cross = 'golden'
            self.dead_cross_active = False  # 金叉解除死叉状态
            return 'golden'
        elif not curr_above and prev_above:
            self.intraday_macd_cross = 'dead'
            self.dead_cross_active = True   # 死叉激活
            return 'dead'

        return None

    def check_volume_price_alert(self, current_time):
        """
        检查量价异常
        返回: (alert_type, details)
        alert_type: None / 'volume_surge' / 'volume_plunge' / 'price_plunge'
        """
        if len(self.minute_volumes) < 5:
            return None, {}

        # 1. 分钟放量检测
        recent_vol_avg = np.mean(self.minute_volumes[-20:]) if len(self.minute_volumes) >= 20 else np.mean(self.minute_volumes)
        current_vol = self.minute_volumes[-1]
        realtime_vol_ratio = current_vol / recent_vol_avg if recent_vol_avg > 0 else 1.0

        # 2. 累计量比检测
        if self.prev_day_volume > 0 and len(self.minute_times) > 0:
            # 计算当前时间占交易时间比例 (假设240分钟交易时间)
            minutes_elapsed = len(self.minute_prices)
            time_ratio = minutes_elapsed / 240.0
            expected_volume = self.prev_day_volume * time_ratio
            cumulative_vol_ratio = self.cumulative_volume / expected_volume if expected_volume > 0 else 1.0
        else:
            cumulative_vol_ratio = 1.0

        # 3. 实时动量检测
        if self.open_price > 0:
            current_momentum = (self.minute_prices[-1] - self.open_price) / self.open_price
        else:
            current_momentum = 0

        details = {
            'realtime_vol_ratio': realtime_vol_ratio,
            'cumulative_vol_ratio': cumulative_vol_ratio,
            'momentum': current_momentum
        }

        # 急跌检测: 动量 < -5%
        if current_momentum < -Config.MOMENTUM_ALERT_THRESHOLD:
            if not self.plunge_detected:
                self.plunge_detected = True
                self.pause_buy_until = current_time + timedelta(minutes=Config.PAUSE_AFTER_PLUNGE_MINUTES)
                return 'price_plunge', details

        # 放量下跌: 量比>3 且 动量<0
        if realtime_vol_ratio > Config.REALTIME_VOL_RATIO_HIGH and current_momentum < -0.01:
            self.pause_buy_until = current_time + timedelta(minutes=10)
            return 'volume_surge_down', details

        # 放量上涨: 量比>3 且 动量>0 (不暂停，但记录)
        if realtime_vol_ratio > Config.REALTIME_VOL_RATIO_HIGH and current_momentum > 0.01:
            return 'volume_surge_up', details

        return None, details

    def check_extreme_market(self, current_price, current_time):
        """
        检查极端行情
        返回: (protection_type, can_buy, can_sell)
        """
        if self.prev_close <= 0:
            return None, True, True

        price_change = (current_price - self.prev_close) / self.prev_close

        # 涨停保护: 停止卖出
        if price_change >= Config.LIMIT_UP_THRESHOLD:
            return 'limit_up', True, False

        # 跌停保护: 停止买入
        if price_change <= Config.LIMIT_DOWN_THRESHOLD:
            return 'limit_down', False, True

        # 大幅波动保护: 5分钟内涨跌幅>3%
        if len(self.recent_prices) >= 5:
            price_5min_ago = self.recent_prices[0]
            volatility_5min = abs(current_price - price_5min_ago) / price_5min_ago
            if volatility_5min > Config.VOLATILITY_PAUSE_THRESHOLD:
                self.pause_all_until = current_time + timedelta(minutes=Config.VOLATILITY_PAUSE_MINUTES)
                self.volatility_pause_count += 1
                return 'high_volatility', False, False

        return None, True, True

    def get_trade_permission(self, current_time):
        """
        获取当前交易许可状态
        返回: (can_buy, can_sell, reason)
        """
        can_buy = True
        can_sell = True
        reasons = []

        # 检查延迟开始
        if self.delayed_start:
            if current_time.hour == 9 and current_time.minute < Config.DELAYED_START_MINUTE:
                return False, False, "开盘跳空%.1f%%，延迟至9:%d" % (self.open_gap * 100, Config.DELAYED_START_MINUTE)

        # 检查全局暂停
        if self.pause_all_until and current_time < self.pause_all_until:
            remaining = (self.pause_all_until - current_time).seconds // 60
            return False, False, "波动暂停，剩余%d分钟" % remaining

        # 检查买入暂停
        if self.pause_buy_until and current_time < self.pause_buy_until:
            can_buy = False
            remaining = (self.pause_buy_until - current_time).seconds // 60
            reasons.append("买入暂停%d分钟" % remaining)

        # 检查分时死叉
        if self.dead_cross_active:
            can_buy = False
            reasons.append("分时死叉")

        # 检查涨跌停
        if not self.can_buy:
            can_buy = False
            reasons.append("跌停保护")
        if not self.can_sell:
            can_sell = False
            reasons.append("涨停保护")

        reason = "; ".join(reasons) if reasons else ""
        return can_buy, can_sell, reason

    def process_bar(self, price, volume, current_time, prev_close=None):
        """
        处理每根K线，更新所有监控状态
        返回: (can_buy, can_sell, alerts)
        """
        alerts = []

        # 设置昨收 (如果提供)
        if prev_close and self.prev_close == 0:
            self.prev_close = prev_close

        # 首次设置开盘价
        if self.open_price == 0:
            has_gap, gap = self.check_open_gap(price)
            if has_gap:
                alerts.append(('open_gap', gap))

        # 更新分钟数据
        self.update_minute_data(price, volume, current_time)

        # 计算分时MACD
        self.calculate_intraday_macd()
        cross = self.check_intraday_macd_cross()
        if cross:
            alerts.append(('macd_cross', cross))

        # 检查量价异常
        vol_alert, vol_details = self.check_volume_price_alert(current_time)
        if vol_alert:
            alerts.append(('volume_price', vol_alert, vol_details))

        # 检查极端行情
        extreme_type, ext_can_buy, ext_can_sell = self.check_extreme_market(price, current_time)
        if extreme_type:
            self.can_buy = ext_can_buy
            self.can_sell = ext_can_sell
            alerts.append(('extreme', extreme_type))

        # 获取最终交易许可
        can_buy, can_sell, reason = self.get_trade_permission(current_time)

        return can_buy, can_sell, alerts, reason


# ==================== 4状态仓位控制器 ====================
class MarketStateController:
    """4状态仓位控制"""

    STRONG_LONG = "STRONG_LONG"
    WEAK_LONG = "WEAK_LONG"
    WEAK_SHORT = "WEAK_SHORT"
    STRONG_SHORT = "STRONG_SHORT"

    # 权重配置
    W_VP_STRONG = 1.5
    W_VP_MODERATE = 0.8
    W_VP_WEAK = 0.3
    W_DOW_CONFIRMED = 1.0
    W_DOW_UNCONFIRMED = 0.3
    W_MACD_EXPAND = 1.2
    W_MACD_GROW = 0.6
    W_MACD_SHRINK = 0.0
    W_MACD_CROSS = 0.8
    W_RSI_STRONG = 0.8
    W_RSI_MODERATE = 0.4
    W_RSI_WEAK = 0.3
    W_RSI_EXTREME = 0.6
    W_DIVERGE = 1.5
    W_DIVERGE_PENALTY = 1.0

    RSI_VERY_STRONG = 70
    RSI_STRONG_LOW = 55
    RSI_WEAK_HIGH = 50
    RSI_WEAK_LOW = 40

    @staticmethod
    def calculate_scores(vol_ratio, momentum, dow_confirmed, dow_trend,
                        macd_hist, macd_hist_prev, rsi,
                        has_top_diverge, has_bottom_diverge,
                        macd_cross=None):
        bull_score = 0.0
        bear_score = 0.0
        details = {}

        # 1. 量价评分
        vp_bull = 0.0
        vp_bear = 0.0
        if vol_ratio > Config.VOL_RATIO_HIGH:
            if momentum > Config.MOM_STRONG_UP:
                vp_bull = MarketStateController.W_VP_STRONG
            elif momentum > 0:
                vp_bull = MarketStateController.W_VP_MODERATE
            elif momentum < Config.MOM_STRONG_DOWN:
                vp_bear = MarketStateController.W_VP_STRONG
            elif momentum < 0:
                vp_bear = MarketStateController.W_VP_MODERATE
        elif vol_ratio < Config.VOL_RATIO_LOW:
            if momentum > 0:
                vp_bull = MarketStateController.W_VP_WEAK
            elif momentum < 0:
                vp_bear = MarketStateController.W_VP_WEAK
        else:
            if momentum > 0:
                vp_bull = 0.5
            elif momentum < 0:
                vp_bear = 0.5
        bull_score += vp_bull
        bear_score += vp_bear
        details['vol_price'] = {'bull': vp_bull, 'bear': vp_bear}

        # 2. 道氏确认评分
        dow_bull = 0.0
        dow_bear = 0.0
        if dow_confirmed:
            if dow_trend == 'UP':
                dow_bull = MarketStateController.W_DOW_CONFIRMED
            elif dow_trend == 'DOWN':
                dow_bear = MarketStateController.W_DOW_CONFIRMED
        else:
            if momentum > 0:
                dow_bull = MarketStateController.W_DOW_UNCONFIRMED
            elif momentum < 0:
                dow_bear = MarketStateController.W_DOW_UNCONFIRMED
        bull_score += dow_bull
        bear_score += dow_bear
        details['dow'] = {'bull': dow_bull, 'bear': dow_bear}

        # 3. MACD柱状图评分
        macd_bull = 0.0
        macd_bear = 0.0
        if macd_hist is not None and macd_hist_prev is not None:
            if macd_cross == 'golden':
                macd_bull = MarketStateController.W_MACD_CROSS
            elif macd_cross == 'dead':
                macd_bear = MarketStateController.W_MACD_CROSS
            elif macd_hist > 0:
                if macd_hist_prev > 0:
                    ratio = macd_hist / macd_hist_prev if macd_hist_prev != 0 else 1
                    if ratio >= Config.MACD_EXPAND_RATIO:
                        macd_bull = MarketStateController.W_MACD_EXPAND
                    elif ratio > 1:
                        macd_bull = MarketStateController.W_MACD_GROW
                else:
                    macd_bull = MarketStateController.W_MACD_GROW
            else:
                if macd_hist_prev < 0:
                    ratio = macd_hist / macd_hist_prev if macd_hist_prev != 0 else 1
                    if ratio >= Config.MACD_EXPAND_RATIO:
                        macd_bear = MarketStateController.W_MACD_EXPAND
                    elif ratio > 1:
                        macd_bear = MarketStateController.W_MACD_GROW
                else:
                    macd_bear = MarketStateController.W_MACD_GROW
        bull_score += macd_bull
        bear_score += macd_bear
        details['macd'] = {'bull': macd_bull, 'bear': macd_bear}

        # 4. RSI评分
        rsi_bull = 0.0
        rsi_bear = 0.0
        if rsi >= Config.RSI_OVERBOUGHT:
            rsi_bear = MarketStateController.W_RSI_EXTREME
        elif rsi >= MarketStateController.RSI_VERY_STRONG:
            rsi_bull = MarketStateController.W_RSI_MODERATE
        elif rsi >= MarketStateController.RSI_STRONG_LOW:
            rsi_bull = MarketStateController.W_RSI_STRONG
        elif rsi >= MarketStateController.RSI_WEAK_HIGH:
            rsi_bull = MarketStateController.W_RSI_MODERATE
        elif rsi >= MarketStateController.RSI_WEAK_LOW:
            rsi_bear = MarketStateController.W_RSI_WEAK
        elif rsi > Config.RSI_OVERSOLD:
            rsi_bear = MarketStateController.W_RSI_MODERATE
        else:
            rsi_bull = MarketStateController.W_RSI_EXTREME
        bull_score += rsi_bull
        bear_score += rsi_bear
        details['rsi'] = {'bull': rsi_bull, 'bear': rsi_bear}

        # 5. 背离评分
        div_bull = 0.0
        div_bear = 0.0
        if has_top_diverge:
            div_bear = MarketStateController.W_DIVERGE
            bull_score = max(0, bull_score - MarketStateController.W_DIVERGE_PENALTY)
        if has_bottom_diverge:
            div_bull = MarketStateController.W_DIVERGE
            bear_score = max(0, bear_score - MarketStateController.W_DIVERGE_PENALTY)
        bull_score += div_bull
        bear_score += div_bear
        details['diverge'] = {'bull': div_bull, 'bear': div_bear}

        return max(0, bull_score), max(0, bear_score), details

    @staticmethod
    def determine_state(bull_score, bear_score):
        diff = bull_score - bear_score
        if diff >= 1.5 and bull_score >= 2.5:
            return MarketStateController.STRONG_LONG
        if diff <= -1.5 and bear_score >= 2.5:
            return MarketStateController.STRONG_SHORT
        if diff > 0:
            return MarketStateController.WEAK_LONG
        return MarketStateController.WEAK_SHORT

    @staticmethod
    def calculate_position(vol_ratio, momentum, dow_confirmed, dow_trend,
                          macd_hist, macd_hist_prev, rsi,
                          has_top_diverge, has_bottom_diverge,
                          macd_cross=None,
                          # v3.1.2 参数
                          dif=None, dea=None, rsi_history=None,
                          k=50, d=50, kdj_cross=None, k_value=50,
                          bias=0, vp_state='normal',
                          # v3.1.3 状态机 (由外部传入)
                          signal_state_machine=None):
        """
        计算目标仓位
        v3.1.3: 使用信号状态机计算系数
          执行仓位 = 基础仓位 × 状态机系数
        """
        bull_score, bear_score, details = MarketStateController.calculate_scores(
            vol_ratio, momentum, dow_confirmed, dow_trend,
            macd_hist, macd_hist_prev, rsi,
            has_top_diverge, has_bottom_diverge,
            macd_cross=macd_cross
        )
        state = MarketStateController.determine_state(bull_score, bear_score)

        # 确定基础仓位
        if state == MarketStateController.STRONG_LONG:
            base_position = Config.POSITION_STRONG_LONG
        elif state == MarketStateController.WEAK_LONG:
            base_position = Config.POSITION_WEAK_LONG
        elif state == MarketStateController.WEAK_SHORT:
            base_position = Config.POSITION_WEAK_SHORT
        else:
            base_position = Config.POSITION_STRONG_SHORT

        # 原有的评分微调
        if state != MarketStateController.STRONG_SHORT:
            score_diff = bull_score - bear_score
            adjustment = min(0.10, max(-0.10, score_diff * 0.02))
            adjusted_position = max(0, min(1.0, base_position + adjustment))
        else:
            adjusted_position = 0.0

        # ===== v3.1.3: 信号得分 + 状态机 =====
        # 1. 计算6类信号总得分
        signal_score, triggered_signals, has_liquidation = SignalScorer.calculate_total_score(
            macd_cross=macd_cross,
            vol_ratio=vol_ratio,
            dif=dif,
            dea=dea,
            rsi=rsi,
            rsi_history=rsi_history,
            top_diverge=has_top_diverge,
            bottom_diverge=has_bottom_diverge,
            k=k,
            d=d,
            kdj_cross=kdj_cross,
            k_value=k_value,
            bias=bias,
            vp_state=vp_state
        )

        # 2. 状态机处理得分，返回系数
        if signal_state_machine is not None:
            signal_coef, transition_desc = signal_state_machine.process(signal_score, has_liquidation)
            machine_state = signal_state_machine.state
            machine_state_cn = signal_state_machine.get_state_name_cn()
        else:
            # 无状态机时，使用简单映射
            signal_coef = 1.0
            transition_desc = None
            machine_state = 'NORMAL'
            machine_state_cn = '正常'

        # 3. 执行仓位 = 基础仓位 × 状态机系数
        final_position = adjusted_position * signal_coef
        final_position = max(0.0, min(1.0, final_position))

        # 添加详情
        details['signal_score'] = signal_score
        details['signal_coef'] = signal_coef
        details['triggered_signals'] = triggered_signals
        details['has_liquidation'] = has_liquidation
        details['machine_state'] = machine_state
        details['machine_state_cn'] = machine_state_cn
        details['transition_desc'] = transition_desc
        details['base_position'] = base_position
        details['adjusted_position'] = adjusted_position

        return state, final_position, bull_score, bear_score, details


# ==================== 盘中交易器 ====================
class IntradayTrader:
    """
    盘中交易器 - 根据市场状态执行分批买卖
    """

    def __init__(self, context):
        self.context = context
        self.reset_daily_state()

    def reset_daily_state(self):
        """重置每日状态"""
        # 通用
        self.open_price = 0              # 开盘价
        self.first_buy_price = 0         # 首次买入价
        self.first_sell_price = 0        # 首次卖出价
        self.last_buy_price = 0          # 最近买入价
        self.last_sell_price = 0         # 最近卖出价

        # 已触发的价格档位 (防止同一档位重复触发)
        self.triggered_buy_levels = set()   # 已触发买入的价格档位
        self.triggered_sell_levels = set()  # 已触发卖出的价格档位

        # 死叉状态
        self.dead_cross_triggered = False   # 当日是否出现死叉

        # 交易统计
        self.today_buy_count = 0
        self.today_sell_count = 0
        self.today_buy_amount = 0        # 当日买入股数

    def set_open_price(self, price):
        """设置开盘价"""
        self.open_price = price

    def check_dead_cross(self, dif, dea):
        """检查死叉状态"""
        if not self.dead_cross_triggered:
            if MACDCalculator.check_dead_cross_state(dif, dea):
                self.dead_cross_triggered = True
                return True
        return False

    def _get_buy_level(self, price, base_price, first_drop, step_drop):
        """计算买入档位编号"""
        if price >= base_price * (1 - first_drop):
            return -1  # 未到首档
        drop_ratio = (base_price - price) / base_price
        if drop_ratio < first_drop:
            return -1
        level = int((drop_ratio - first_drop) / step_drop) + 1
        return level

    def _get_sell_level(self, price, base_price, first_rise, step_rise):
        """计算卖出档位编号"""
        if price <= base_price * (1 + first_rise):
            return -1  # 未到首档
        rise_ratio = (price - base_price) / base_price
        if rise_ratio < first_rise:
            return -1
        level = int((rise_ratio - first_rise) / step_rise) + 1
        return level

    def _calculate_trade_amount(self, total_value, current_price, ratio):
        """计算交易股数 (100股整数倍)"""
        target_value = total_value * ratio
        shares = int(target_value / current_price / 100) * 100
        return shares

    # ==================== STRONG_LONG 逻辑 ====================
    def handle_strong_long(self, security, current_price, total_value, position_ratio,
                          current_shares, closeable_shares, dif, dea,
                          can_buy=True, can_sell=True):
        """
        强多头盘中交易:
        1. 回落买入: 开盘价-1%起，每-1%买10%
        2. 死叉卖出: 持仓>100%时清可卖仓
        3. 反弹卖出: 首次买入价+2%起，每+1%卖10%

        新增: can_buy/can_sell 由 IntradayMonitor 控制
        """
        context = self.context
        traded = False

        # 检查死叉 (日线MACD，保留兼容)
        if self.check_dead_cross(dif, dea):
            # 持仓超过100%时，死叉清仓 (死叉清仓不受can_sell限制)
            if position_ratio > 1.0 and closeable_shares >= 100:
                order(security, -closeable_shares)
                log.warning("[STRONG_LONG 死叉清仓] 卖出: %d股" % closeable_shares)
                self.today_sell_count += 1
                return True

        # 1. 回落买入逻辑 (受can_buy限制)
        if self.open_price > 0 and can_buy:
            buy_level = self._get_buy_level(
                current_price, self.open_price,
                Config.SL_BUY_DROP_FIRST, Config.SL_BUY_DROP_STEP
            )

            if buy_level >= 1 and buy_level not in self.triggered_buy_levels:
                # 检查仓位上限
                if position_ratio < Config.SL_MAX_POSITION:
                    buy_shares = self._calculate_trade_amount(
                        total_value, current_price, Config.TRADE_RATIO
                    )
                    # 检查可用资金
                    cash = context.portfolio.cash
                    max_buy = int(cash * 0.98 / current_price / 100) * 100
                    buy_shares = min(buy_shares, max_buy)

                    if buy_shares >= 100:
                        order(security, buy_shares)
                        self.triggered_buy_levels.add(buy_level)
                        self.today_buy_count += 1
                        self.today_buy_amount += buy_shares

                        if self.first_buy_price == 0:
                            self.first_buy_price = current_price
                        self.last_buy_price = current_price

                        log.info("[STRONG_LONG 回落买入] 档位%d 价格%.2f 买入%d股 仓位%.1f%%" %
                                (buy_level, current_price, buy_shares, position_ratio * 100))
                        traded = True

        # 2. 反弹卖出逻辑 (需先有买入，受can_sell限制)
        if self.first_buy_price > 0 and not self.dead_cross_triggered and can_sell:
            sell_level = self._get_sell_level(
                current_price, self.first_buy_price,
                Config.SL_SELL_RISE_FIRST, Config.SL_SELL_RISE_STEP
            )

            if sell_level >= 1 and sell_level not in self.triggered_sell_levels:
                if closeable_shares >= 100:
                    sell_shares = self._calculate_trade_amount(
                        total_value, current_price, Config.TRADE_RATIO
                    )
                    sell_shares = min(sell_shares, closeable_shares)
                    sell_shares = int(sell_shares / 100) * 100

                    if sell_shares >= 100:
                        order(security, -sell_shares)
                        self.triggered_sell_levels.add(sell_level)
                        self.today_sell_count += 1

                        if self.first_sell_price == 0:
                            self.first_sell_price = current_price
                        self.last_sell_price = current_price

                        log.info("[STRONG_LONG 反弹卖出] 档位%d 价格%.2f 卖出%d股" %
                                (sell_level, current_price, sell_shares))
                        traded = True

        return traded

    # ==================== WEAK_LONG 逻辑 ====================
    def handle_weak_long(self, security, current_price, total_value, position_ratio,
                        current_shares, closeable_shares, dif, dea,
                        can_buy=True, can_sell=True):
        """
        弱多头盘中交易:
        1. 上涨卖出: 开盘价+2%起，每+1%卖10%
        2. 回落买入: 首次卖出价-1%起，每-1%买10%，到60%阈值
        3. 死叉后停止买入

        新增: can_buy/can_sell 由 IntradayMonitor 控制
        """
        context = self.context
        traded = False

        # 检查死叉 (日线MACD，保留兼容)
        self.check_dead_cross(dif, dea)

        # 1. 上涨卖出 (受can_sell限制)
        if self.open_price > 0 and closeable_shares >= 100 and can_sell:
            sell_level = self._get_sell_level(
                current_price, self.open_price,
                Config.WL_SELL_RISE_FIRST, Config.WL_SELL_RISE_STEP
            )

            if sell_level >= 1 and sell_level not in self.triggered_sell_levels:
                sell_shares = self._calculate_trade_amount(
                    total_value, current_price, Config.TRADE_RATIO
                )
                sell_shares = min(sell_shares, closeable_shares)
                sell_shares = int(sell_shares / 100) * 100

                if sell_shares >= 100:
                    order(security, -sell_shares)
                    self.triggered_sell_levels.add(sell_level)
                    self.today_sell_count += 1

                    if self.first_sell_price == 0:
                        self.first_sell_price = current_price
                    self.last_sell_price = current_price

                    log.info("[WEAK_LONG 上涨卖出] 档位%d 价格%.2f 卖出%d股" %
                            (sell_level, current_price, sell_shares))
                    traded = True

        # 2. 回落买入 (需先有卖出，且未死叉，受can_buy限制)
        if self.first_sell_price > 0 and not self.dead_cross_triggered and can_buy:
            buy_level = self._get_buy_level(
                current_price, self.first_sell_price,
                Config.WL_BUY_DROP_FIRST, Config.WL_BUY_DROP_STEP
            )

            if buy_level >= 1 and buy_level not in self.triggered_buy_levels:
                # 检查仓位阈值
                if position_ratio < Config.POSITION_WEAK_LONG:
                    buy_shares = self._calculate_trade_amount(
                        total_value, current_price, Config.TRADE_RATIO
                    )
                    cash = context.portfolio.cash
                    max_buy = int(cash * 0.98 / current_price / 100) * 100
                    buy_shares = min(buy_shares, max_buy)

                    if buy_shares >= 100:
                        order(security, buy_shares)
                        self.triggered_buy_levels.add(buy_level)
                        self.today_buy_count += 1
                        self.today_buy_amount += buy_shares

                        if self.first_buy_price == 0:
                            self.first_buy_price = current_price
                        self.last_buy_price = current_price

                        log.info("[WEAK_LONG 回落买入] 档位%d 价格%.2f 买入%d股 仓位%.1f%%" %
                                (buy_level, current_price, buy_shares, position_ratio * 100))
                        traded = True

        return traded

    # ==================== WEAK_SHORT 逻辑 ====================
    def handle_weak_short(self, security, current_price, total_value, position_ratio,
                         current_shares, closeable_shares, dif, dea,
                         can_buy=True, can_sell=True):
        """
        弱空头盘中交易:
        1. 上涨卖出: 开盘价+2%卖20%，之后每+1%卖10%
        2. 回落买入: 首次卖出价-1%起，每-1%买10%，到30%阈值
        3. 死叉后停止买入

        新增: can_buy/can_sell 由 IntradayMonitor 控制
        """
        context = self.context
        traded = False

        # 检查死叉 (日线MACD，保留兼容)
        self.check_dead_cross(dif, dea)

        # 1. 上涨卖出 (受can_sell限制)
        if self.open_price > 0 and closeable_shares >= 100 and can_sell:
            sell_level = self._get_sell_level(
                current_price, self.open_price,
                Config.WS_SELL_RISE_FIRST, Config.WS_SELL_RISE_STEP
            )

            if sell_level >= 1 and sell_level not in self.triggered_sell_levels:
                # 首次卖20%，后续卖10%
                if sell_level == 1:
                    sell_ratio = Config.WS_FIRST_SELL_RATIO
                else:
                    sell_ratio = Config.WS_SELL_RATIO

                sell_shares = self._calculate_trade_amount(
                    total_value, current_price, sell_ratio
                )
                sell_shares = min(sell_shares, closeable_shares)
                sell_shares = int(sell_shares / 100) * 100

                if sell_shares >= 100:
                    order(security, -sell_shares)
                    self.triggered_sell_levels.add(sell_level)
                    self.today_sell_count += 1

                    if self.first_sell_price == 0:
                        self.first_sell_price = current_price
                    self.last_sell_price = current_price

                    log.info("[WEAK_SHORT 上涨卖出] 档位%d 价格%.2f 卖出%d股 (%.0f%%)" %
                            (sell_level, current_price, sell_shares, sell_ratio * 100))
                    traded = True

        # 2. 回落买入 (需先有卖出，且未死叉，受can_buy限制)
        if self.first_sell_price > 0 and not self.dead_cross_triggered and can_buy:
            buy_level = self._get_buy_level(
                current_price, self.first_sell_price,
                Config.WS_BUY_DROP_FIRST, Config.WS_BUY_DROP_STEP
            )

            if buy_level >= 1 and buy_level not in self.triggered_buy_levels:
                # 检查仓位阈值
                if position_ratio < Config.POSITION_WEAK_SHORT:
                    buy_shares = self._calculate_trade_amount(
                        total_value, current_price, Config.TRADE_RATIO
                    )
                    cash = context.portfolio.cash
                    max_buy = int(cash * 0.98 / current_price / 100) * 100
                    buy_shares = min(buy_shares, max_buy)

                    if buy_shares >= 100:
                        order(security, buy_shares)
                        self.triggered_buy_levels.add(buy_level)
                        self.today_buy_count += 1
                        self.today_buy_amount += buy_shares

                        if self.first_buy_price == 0:
                            self.first_buy_price = current_price
                        self.last_buy_price = current_price

                        log.info("[WEAK_SHORT 回落买入] 档位%d 价格%.2f 买入%d股 仓位%.1f%%" %
                                (buy_level, current_price, buy_shares, position_ratio * 100))
                        traded = True

        return traded

    def handle_strong_short(self, security, current_price, total_value, position_ratio,
                           current_shares, closeable_shares):
        """
        强空头: 清仓，不参与盘中交易
        """
        # 清仓逻辑由主策略处理
        return False


# ==================== 风险管理器 ====================
class RiskManager:
    """
    风险管理器 (v3.1.4 简化版)

    两种风控机制:
    1. 盘中极端保护: 日内高点回落超过12%，立即清仓
    2. 账户止损保护: 连续3天亏损且累计亏损超过8%，清仓

    统一5天冷却期，与状态机衔接
    """

    def __init__(self, context):
        self.context = context
        # 日内最高价追踪
        self.intraday_high = 0
        # 账户每日净值记录 (用于连续亏损判断)
        self.daily_values = []  # [(date, value), ...]
        self.initial_value = 0  # 记录起始净值

    def reset_daily(self):
        """每日重置盘中最高价"""
        self.intraday_high = 0

    def update_intraday_high(self, current_price):
        """更新日内最高价"""
        if current_price > self.intraday_high:
            self.intraday_high = current_price
            return True
        return False

    def check_intraday_drop(self, current_price):
        """
        检查日内高点回落
        返回: (是否触发, 回落比例)
        """
        if self.intraday_high <= 0:
            return False, 0
        drop_ratio = (self.intraday_high - current_price) / self.intraday_high
        if drop_ratio >= Config.INTRADAY_DROP_THRESHOLD:
            return True, drop_ratio
        return False, drop_ratio

    def record_daily_value(self, current_date, total_value):
        """
        记录每日净值 (盘后调用)
        """
        # 初始化起始净值
        if self.initial_value == 0:
            self.initial_value = total_value

        # 添加今日记录
        self.daily_values.append((current_date, total_value))

        # 只保留最近10天记录
        if len(self.daily_values) > 10:
            self.daily_values = self.daily_values[-10:]

    def check_account_stop_loss(self):
        """
        检查账户止损条件
        条件: 连续3天亏损 且 累计亏损超过8%
        返回: (是否触发, 连续亏损天数, 累计亏损比例)
        """
        if len(self.daily_values) < Config.ACCOUNT_LOSS_DAYS:
            return False, 0, 0

        # 取最近N天
        recent = self.daily_values[-Config.ACCOUNT_LOSS_DAYS:]

        # 检查是否连续亏损
        consecutive_loss = True
        for i in range(1, len(recent)):
            if recent[i][1] >= recent[i-1][1]:
                consecutive_loss = False
                break

        if not consecutive_loss:
            return False, 0, 0

        # 计算累计亏损 (相对于N天前)
        start_value = recent[0][1]
        end_value = recent[-1][1]
        loss_ratio = (start_value - end_value) / start_value if start_value > 0 else 0

        if loss_ratio >= Config.ACCOUNT_LOSS_THRESHOLD:
            return True, Config.ACCOUNT_LOSS_DAYS, loss_ratio

        return False, Config.ACCOUNT_LOSS_DAYS, loss_ratio

    def is_in_cooldown(self, current_date):
        """检查是否在冷却期"""
        if self.context.cooldown_end_date is None:
            return False
        return current_date < self.context.cooldown_end_date

    def enter_cooldown(self, current_date, reason='unknown'):
        """
        进入冷却期
        reason: 'intraday_drop' 或 'account_loss'
        """
        self.context.cooldown_end_date = current_date + timedelta(days=Config.COOLDOWN_DAYS)
        self.context.cooldown_reason = reason
        # 重置日内高点
        self.intraday_high = 0
        log.warning("[风控] 进入%d天冷却期，原因: %s，结束日期: %s" %
                   (Config.COOLDOWN_DAYS, reason, self.context.cooldown_end_date))
        return self.context.cooldown_end_date

    def exit_cooldown(self):
        """退出冷却期"""
        self.context.cooldown_end_date = None
        self.context.cooldown_reason = None
        log.info("[风控] 冷却期结束")

    def execute_emergency_liquidation(self, security, reason, drop_ratio=0, loss_ratio=0):
        """
        执行紧急清仓
        """
        position = self.context.portfolio.positions.get(security)
        if position is None or position.amount <= 0:
            return False

        closeable = position.closeable_amount
        if closeable < 100:
            return False

        order(security, -closeable)

        if reason == 'intraday_drop':
            log.warning("[盘中清仓] 日内高点回落 %.1f%% >= %.1f%%，卖出 %d 股" %
                       (drop_ratio * 100, Config.INTRADAY_DROP_THRESHOLD * 100, closeable))
        elif reason == 'account_loss':
            log.warning("[账户止损] 连续%d天亏损，累计亏损 %.1f%% >= %.1f%%，卖出 %d 股" %
                       (Config.ACCOUNT_LOSS_DAYS, loss_ratio * 100,
                        Config.ACCOUNT_LOSS_THRESHOLD * 100, closeable))

        return True


# ==================== 策略主控制器 ====================
class Strategy:
    """策略主控制器"""

    def __init__(self, context):
        self.context = context
        self.risk_mgr = RiskManager(context)
        self.intraday_trader = IntradayTrader(context)
        self.intraday_monitor = IntradayMonitor()  # 新增: 盘中实时监控器
        self._logged_today = set()

    def _log_once(self, key, level, msg):
        if key not in self._logged_today:
            self._logged_today.add(key)
            if level == 'info':
                log.info(msg)
            elif level == 'warning':
                log.warning(msg)

    def _reset_daily_log(self):
        self._logged_today = set()

    def reset_daily(self):
        """每日重置 (供before_trading_start调用)"""
        self._reset_daily_log()
        self.intraday_trader.reset_daily_state()
        self.intraday_monitor.reset_daily()
        self.risk_mgr.reset_daily()  # v3.1.4: 重置日内最高价

    def on_bar(self, data):
        """盘中处理 - 集成实时监控"""
        context = self.context
        security = context.stock

        if security not in data:
            return
        bar = data[security]
        current_price = bar.close
        current_volume = bar.volume if hasattr(bar, 'volume') else 0

        if current_price <= 0:
            return

        current_date = context.blotter.current_dt.date()
        current_time = context.blotter.current_dt

        # 设置开盘价 (首次获取)
        if self.intraday_trader.open_price == 0:
            self.intraday_trader.set_open_price(bar.open if hasattr(bar, 'open') else current_price)

        # ===== 盘中监控器处理 =====
        # 设置昨收和昨日成交量 (首次)
        if self.intraday_monitor.prev_close == 0:
            self.intraday_monitor.set_prev_close(context.prev_close)
        if self.intraday_monitor.prev_day_volume == 0:
            self.intraday_monitor.set_prev_day_volume(getattr(context, 'prev_day_volume', 0))

        # 处理当前K线，获取交易许可
        can_buy, can_sell, alerts, reason = self.intraday_monitor.process_bar(
            current_price, current_volume, current_time, context.prev_close
        )

        # 记录预警日志
        for alert in alerts:
            alert_type = alert[0]
            if alert_type == 'open_gap':
                self._log_once('open_gap', 'warning', "[开盘跳空] %.1f%%" % (alert[1] * 100))
            elif alert_type == 'macd_cross':
                cross_type = alert[1]
                if cross_type == 'dead':
                    self._log_once('intraday_dead', 'warning', "[分时死叉] 停止买入")
                else:
                    self._log_once('intraday_golden', 'info', "[分时金叉] 恢复买入")
            elif alert_type == 'volume_price':
                vol_alert = alert[1]
                if vol_alert == 'price_plunge':
                    self._log_once('plunge', 'warning', "[急跌预警] 动量%.1f%% 暂停买入30分钟" %
                                  (alert[2].get('momentum', 0) * 100))
                elif vol_alert == 'volume_surge_down':
                    self._log_once('vol_surge_down', 'warning', "[放量下跌] 量比%.1f 暂停买入" %
                                  alert[2].get('realtime_vol_ratio', 0))
            elif alert_type == 'extreme':
                extreme_type = alert[1]
                if extreme_type == 'limit_up':
                    self._log_once('limit_up', 'info', "[涨停保护] 停止卖出")
                elif extreme_type == 'limit_down':
                    self._log_once('limit_down', 'warning', "[跌停保护] 停止买入")
                elif extreme_type == 'high_volatility':
                    self._log_once('volatility', 'warning', "[波动暂停] 5分钟内波动过大")

        # 如果有保护原因，记录日志
        if reason and not can_buy and not can_sell:
            self._log_once('protection_' + str(current_time.minute), 'info', "[保护状态] %s" % reason)

        # 交易时间检查 (在监控器处理之后)
        if current_time.hour < Config.TRADE_START_HOUR or \
           (current_time.hour == Config.TRADE_START_HOUR and current_time.minute < Config.TRADE_START_MINUTE):
            return

        # 冷却期检查
        if self.risk_mgr.is_in_cooldown(current_date):
            remaining = (context.cooldown_end_date - current_date).days
            self._log_once('cooldown', 'warning', "[冷却期] 剩余%d天，交易暂停" % remaining)
            return

        # 获取持仓信息
        total_value = context.portfolio.total_value
        if total_value > context.peak_value:
            context.peak_value = total_value

        position = context.portfolio.positions.get(security)
        current_shares = position.amount if position else 0
        closeable_shares = max(0, current_shares - self.intraday_trader.today_buy_amount)
        current_value = current_shares * current_price
        position_ratio = Utils.safe_divide(current_value, total_value)

        # ===== v3.1.4: 日内高点回落检测 =====
        # 更新日内最高价
        self.risk_mgr.update_intraday_high(current_price)

        # 检查是否触发日内极端保护
        drop_triggered, drop_ratio = self.risk_mgr.check_intraday_drop(current_price)
        if drop_triggered and closeable_shares >= 100:
            # 执行紧急清仓
            self.risk_mgr.execute_emergency_liquidation(
                security, 'intraday_drop', drop_ratio=drop_ratio)
            self.risk_mgr.enter_cooldown(current_date, reason='intraday_drop')
            # 强制状态机进入清仓状态
            if hasattr(context, 'signal_state_machine'):
                context.signal_state_machine.force_liquidation()
            return

        # 获取日线MACD用于死叉检测 (保留兼容)
        dif = getattr(context, 'macd_dif', None)
        dea = getattr(context, 'macd_dea', None)

        # 根据市场状态执行盘中交易 (传入监控器的can_buy/can_sell)
        market_state = getattr(context, 'market_state', MarketStateController.WEAK_LONG)
        target_position = getattr(context, 'target_position', 0.5)

        if market_state == MarketStateController.STRONG_LONG or target_position < 0.8:
            # 强多头 或 目标仓位<80% 都执行加仓逻辑
            self.intraday_trader.handle_strong_long(
                security, current_price, total_value, position_ratio,
                current_shares, closeable_shares, dif, dea,
                can_buy=can_buy, can_sell=can_sell
            )

        elif market_state == MarketStateController.WEAK_LONG:
            self.intraday_trader.handle_weak_long(
                security, current_price, total_value, position_ratio,
                current_shares, closeable_shares, dif, dea,
                can_buy=can_buy, can_sell=can_sell
            )

        elif market_state == MarketStateController.WEAK_SHORT:
            self.intraday_trader.handle_weak_short(
                security, current_price, total_value, position_ratio,
                current_shares, closeable_shares, dif, dea,
                can_buy=can_buy, can_sell=can_sell
            )

        # STRONG_SHORT: 不参与盘中交易，由收盘调仓处理

        # 收盘前检查是否需要调整到目标仓位
        if current_time.hour == Config.TRADE_END_HOUR and current_time.minute >= Config.TRADE_END_MINUTE:
            self._execute_close_adjustment(security, current_price, position_ratio,
                                          current_shares, closeable_shares, total_value, current_date)

    def _execute_close_adjustment(self, security, current_price, position_ratio,
                                  current_shares, closeable_shares, total_value, current_date):
        """收盘前调仓 (主要处理STRONG_SHORT清仓)"""
        context = self.context

        if context.close_adjusted_today:
            return

        market_state = getattr(context, 'market_state', MarketStateController.WEAK_LONG)

        # STRONG_SHORT 需要清仓
        if market_state == MarketStateController.STRONG_SHORT:
            if closeable_shares >= 100:
                order(security, -closeable_shares)
                context.close_adjusted_today = True
                log.info("[STRONG_SHORT 清仓] 卖出%d股" % closeable_shares)


# ==================== PTrade 入口函数 ====================

def initialize(context):
    """初始化"""
    context.stock = Config.STOCK
    set_benchmark(Config.BENCHMARK)
    set_universe([context.stock])

    set_commission(commission_ratio=0.0003, min_commission=5.0)
    set_slippage(slippage=0.002)

    # 状态变量
    context.market_state = MarketStateController.WEAK_LONG
    context.target_position = 0.5
    context.bull_score = 0
    context.bear_score = 0
    context.volatility = 0.5
    context.macd_dif = None
    context.macd_dea = None

    # 风控变量 (v3.1.4 简化版)
    context.peak_value = 0
    context.cooldown_end_date = None
    context.cooldown_reason = None  # 'intraday_drop' 或 'account_loss'

    context.prev_close = 0
    context.prev_day_volume = 0
    context.close_adjusted_today = False

    # v3.1.3: 信号状态机 (跨日持久化)
    context.signal_state_machine = SignalStateMachine()

    # 初始化策略
    context.strategy = Strategy(context)

    log.info("=" * 60)
    log.info("StrategyThr v3.1.4 初始化完成 (简化风控版)")
    log.info("标的: %s" % context.stock)
    log.info("-" * 40)
    log.info("仓位控制 (两层体系):")
    log.info("  第1层: 市场状态评分 → 基础仓位 (90%%/60%%/30%%/0%%)")
    log.info("  第2层: 信号状态机 → 系数 (0~1.5)")
    log.info("  执行仓位 = 基础仓位 × 状态机系数")
    log.info("-" * 40)
    log.info("风控保护 (v3.1.4):")
    log.info("  盘中极端: 日内高点回落>=%.0f%% → 清仓" % (Config.INTRADAY_DROP_THRESHOLD * 100))
    log.info("  账户止损: 连续%d天亏损且>=%.0f%% → 清仓" %
            (Config.ACCOUNT_LOSS_DAYS, Config.ACCOUNT_LOSS_THRESHOLD * 100))
    log.info("  统一冷却: %d天" % Config.COOLDOWN_DAYS)
    log.info("-" * 40)
    log.info("状态机: NORMAL → ALERT → LIQUIDATION → RECOVERY → NORMAL")
    log.info("=" * 60)


def before_trading_start(context, data):
    """盘前分析"""
    # 重置每日状态
    context.close_adjusted_today = False

    if hasattr(context, 'strategy'):
        # 使用新的统一重置方法
        context.strategy.reset_daily()

    try:
        current_date = context.blotter.current_dt.date()

        hist = get_history(Config.TREND_LOOKBACK, '1d', ['close', 'volume'],
                          context.stock, fq='pre', include=False)
        if hist is None or len(hist) < 60:
            log.warning("历史数据不足")
            return

        price_valid, close_prices, _ = Utils.validate_prices(hist['close'].values, min_length=60)
        vol_valid, volumes, _ = Utils.validate_volumes(hist['volume'].values, min_length=20)

        if not price_valid or not vol_valid:
            log.warning("数据验证失败")
            return

        context.prev_close = close_prices[-1]
        context.prev_day_volume = volumes[-1]  # 新增: 保存昨日成交量
        context.volatility = Utils.calculate_volatility(close_prices)

        # 量价分析
        vol_ma = np.mean(volumes[-Config.VOLUME_MA_PERIOD:])
        vol_ratio = Utils.safe_divide(volumes[-1], vol_ma, 1.0)
        momentum = Utils.safe_divide(
            close_prices[-1] - close_prices[-Config.PRICE_MOMENTUM_PERIOD - 1],
            close_prices[-Config.PRICE_MOMENTUM_PERIOD - 1]
        )

        # 道氏确认
        dow_confirmed, dow_trend, _ = DowTheoryAnalyzer.analyze(close_prices)

        # MACD
        dif, dea, hist_macd = MACDCalculator.calculate(close_prices)
        context.macd_dif = dif
        context.macd_dea = dea
        macd_hist = hist_macd[-1] if hist_macd is not None and len(hist_macd) > 0 else 0
        macd_hist_prev = hist_macd[-2] if hist_macd is not None and len(hist_macd) > 1 else 0
        macd_cross = MACDCalculator.detect_cross(dif, dea)

        # RSI
        rsi = RSICalculator.calculate(close_prices)

        # RSI历史 (用于检测钝化和穿越)
        rsi_history = []
        for i in range(min(5, len(close_prices) - Config.RSI_PERIOD)):
            rsi_val = RSICalculator.calculate(close_prices[:-(i) if i > 0 else len(close_prices)])
            rsi_history.insert(0, rsi_val)
        if len(rsi_history) == 0:
            rsi_history = [rsi]

        # 背离检测
        top_diverge, bottom_diverge = DivergenceDetector.detect(close_prices, hist_macd)

        # ===== v3.1.2 新增指标 =====
        # KDJ (需要高低价，这里用收盘价近似)
        k, d, j, k_series, d_series = KDJCalculator.calculate(close_prices)
        kdj_cross, k_value = KDJCalculator.detect_cross(k_series, d_series)

        # BIAS乖离率
        bias = BIASCalculator.calculate(close_prices)

        # 量价偏离度
        vp_state, vp_details = VolumePriceDivergence.analyze(close_prices, volumes)

        # 计算市场状态 (传入所有新增参数 + 状态机)
        market_state, target_position, bull_score, bear_score, score_details = \
            MarketStateController.calculate_position(
                vol_ratio, momentum, dow_confirmed, dow_trend,
                macd_hist, macd_hist_prev, rsi,
                top_diverge, bottom_diverge,
                macd_cross=macd_cross,
                # v3.1.2 新增参数
                dif=dif, dea=dea, rsi_history=rsi_history,
                k=k, d=d, kdj_cross=kdj_cross, k_value=k_value,
                bias=bias, vp_state=vp_state,
                # v3.1.3 状态机
                signal_state_machine=context.signal_state_machine
            )

        context.market_state = market_state
        context.target_position = target_position
        context.bull_score = bull_score
        context.bear_score = bear_score

        # 提取系数信息 (v3.1.3 状态机版)
        signal_coef = score_details.get('signal_coef', 1.0)
        signal_score = score_details.get('signal_score', 0)
        triggered_signals = score_details.get('triggered_signals', [])
        has_liquidation = score_details.get('has_liquidation', False)
        machine_state = score_details.get('machine_state', 'NORMAL')
        machine_state_cn = score_details.get('machine_state_cn', '正常')
        transition_desc = score_details.get('transition_desc', None)
        base_pos = score_details.get('base_position', target_position)
        adj_pos = score_details.get('adjusted_position', target_position)

        # 日志
        log.info("-" * 50)
        log.info("日期: %s" % str(current_date))
        log.info("[市场状态] %s | 目标仓位: %.0f%%" % (market_state, target_position * 100))
        log.info("[得分] 多头: %.1f | 空头: %.1f | 差值: %.1f" %
                (bull_score, bear_score, bull_score - bear_score))
        log.info("[指标] 量比: %.2f | 动量: %.2f%% | RSI: %.1f | 道氏: %s/%s" %
                (vol_ratio, momentum * 100, rsi, dow_confirmed, dow_trend))

        # v3.1.2 新增日志
        log.info("[KDJ] K:%.1f D:%.1f J:%.1f %s" %
                (k, d, j, "金叉" if kdj_cross == 'golden' else ("死叉" if kdj_cross == 'dead' else "")))
        log.info("[BIAS] %.2f%% | [量价] %s" % (bias * 100, vp_state))

        # v3.1.3 状态机信息
        log.info("[信号] 总分: %d | 状态: %s | 系数: %.2f" %
                (signal_score, machine_state_cn, signal_coef))
        if triggered_signals:
            log.info("[触发] %s" % ", ".join(triggered_signals))
        if has_liquidation:
            log.warning("[清仓信号] 触发清仓!")
        if transition_desc:
            log.info("[状态转换] %s" % transition_desc)
        if signal_coef != 1.0:
            log.info("[仓位计算] 基础%.0f%% × %.2f = 执行%.0f%%" %
                    (adj_pos * 100, signal_coef, target_position * 100))

        if macd_cross:
            log.info("[MACD] %s" % ("金叉" if macd_cross == 'golden' else "死叉"))
        if top_diverge:
            log.warning("[背离] 顶背离!")
        if bottom_diverge:
            log.info("[背离] 底背离!")

        # ===== v3.1.4: 简化冷却期管理 =====
        risk_mgr = context.strategy.risk_mgr

        if context.cooldown_end_date:
            if current_date >= context.cooldown_end_date:
                # 冷却期结束，直接恢复交易
                # 状态机会自动从LIQUIDATION通过正常逻辑恢复
                log.info("[冷却期结束] 恢复交易，状态机: %s" %
                        context.signal_state_machine.get_state_name_cn())
                risk_mgr.exit_cooldown()
            else:
                remaining = (context.cooldown_end_date - current_date).days
                reason = getattr(context, 'cooldown_reason', 'unknown')
                log.warning("[冷却期] 剩余%d天 (原因: %s)" % (remaining, reason))

    except Exception as e:
        log.error("[before_trading_start异常] %s" % str(e))


def handle_data(context, data):
    """盘中处理"""
    try:
        context.strategy.on_bar(data)
    except Exception as e:
        log.error("[handle_data异常] %s" % str(e))


def after_trading_end(context, data):
    """盘后处理"""
    security = context.stock
    position = context.portfolio.positions.get(security)
    current_shares = position.amount if position else 0
    current_date = context.blotter.current_dt.date()

    total_value = context.portfolio.total_value
    position_ratio = Utils.safe_divide(current_shares * context.prev_close, total_value)

    drawdown = (context.peak_value - total_value) / context.peak_value if context.peak_value > 0 else 0

    # 盘中交易统计
    trader = context.strategy.intraday_trader
    monitor = context.strategy.intraday_monitor
    risk_mgr = context.strategy.risk_mgr

    log.info("[收盘] 持仓: %d股 | 仓位: %.1f%% | 资产: %.2f | 回撤: %.1f%%" %
            (current_shares, position_ratio * 100, total_value, drawdown * 100))
    log.info("[盘中统计] 买入%d次/%d股 | 卖出%d次 | 状态: %s" %
            (trader.today_buy_count, trader.today_buy_amount,
             trader.today_sell_count, context.market_state))

    # ===== v3.1.4: 账户止损检查 =====
    # 记录每日净值
    risk_mgr.record_daily_value(current_date, total_value)

    # 检查账户止损条件 (连续3天亏损且累计>8%)
    if not risk_mgr.is_in_cooldown(current_date):
        stop_triggered, loss_days, loss_ratio = risk_mgr.check_account_stop_loss()
        if stop_triggered:
            # 执行清仓
            risk_mgr.execute_emergency_liquidation(
                security, 'account_loss', loss_ratio=loss_ratio)
            risk_mgr.enter_cooldown(current_date, reason='account_loss')
            # 强制状态机进入清仓状态
            if hasattr(context, 'signal_state_machine'):
                context.signal_state_machine.force_liquidation()

    # 日内最高回落统计
    if risk_mgr.intraday_high > 0:
        intraday_drop = (risk_mgr.intraday_high - context.prev_close) / risk_mgr.intraday_high
        log.info("[日内] 最高价: %.2f | 回落: %.1f%%" %
                (risk_mgr.intraday_high, intraday_drop * 100))

    # 监控器统计
    monitor_info = []
    if monitor.dead_cross_active:
        monitor_info.append("分时死叉")
    if monitor.plunge_detected:
        monitor_info.append("急跌")
    if monitor.volatility_pause_count > 0:
        monitor_info.append("波动暂停%d次" % monitor.volatility_pause_count)
    if monitor.delayed_start:
        monitor_info.append("跳空延迟")

    if monitor_info:
        log.info("[监控统计] %s | 分钟数据: %d根" % (", ".join(monitor_info), len(monitor.minute_prices)))
    else:
        log.info("[监控统计] 正常 | 分钟数据: %d根" % len(monitor.minute_prices))

    # 状态机状态
    if hasattr(context, 'signal_state_machine'):
        sm = context.signal_state_machine
        log.info("[状态机] %s | 连续强多: %d天 | 清仓天数: %d" %
                (sm.get_state_name_cn(), sm.consecutive_strong_bull_days, sm.days_in_liquidation))
