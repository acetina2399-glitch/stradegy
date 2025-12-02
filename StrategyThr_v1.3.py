# StrategyThr v1.3 - 趋势×量价状态×信号 三维仓位控制策略
# 适用平台: PTrade
# 标的: 603305.SS (旭升集团)

import numpy as np
from datetime import datetime, timedelta

# ==================== 全局参数配置 ====================
class Config:
    """策略参数配置"""
    STOCK = '603305.SS'
    BENCHMARK = '603305.SS'

    # 趋势分析参数
    TREND_MA_SHORT = 20
    TREND_MA_MID = 60
    TREND_MA_LONG = 120
    TREND_SLOPE_THRESHOLD = 0.001
    TREND_LOOKBACK = 250

    # 量价分析参数
    VOLUME_MA_PERIOD = 20
    VOLUME_RATIO_VERY_HIGH = 1.5   # 明显放量
    VOLUME_RATIO_HIGH = 1.2        # 正常偏高
    VOLUME_RATIO_LOW = 0.8         # 缩量
    PRICE_MOMENTUM_SHORT = 5       # 短期动量
    PRICE_MOMENTUM_LONG = 10       # 长期动量
    MOMENTUM_STRONG = 0.02         # 强势动量阈值 2%
    MOMENTUM_WEAK = 0.01           # 弱势动量阈值 1%

    # 信号分析参数
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    MA_SHORT = 5
    MA_LONG = 20                   # 改为20日均线
    SIGNAL_LOOKBACK = 63
    DIVERGENCE_LOOKBACK = 30       # 背离检测回溯天数

    # 仓位控制参数
    STOP_LOSS_THRESHOLD = 0.12       # 止损阈值12%
    MAX_DRAWDOWN = 0.15              # 账户净值回撤15%清仓
    POSITION_CHANGE_THRESHOLD = 0.05
    COOLDOWN_DAYS = 3                # 止损冷却期3天
    DRAWDOWN_COOLDOWN_DAYS = 1       # 账户回撤冷却期1天

    # T+0参数
    T0_SELL_THRESHOLD = 0.06
    T0_REBUY_THRESHOLD = 0.025
    T0_SELL_RATIO = 0.6                # 100%卖出
    T0_COOLDOWN_MINUTES = 5
    T0_POSITION_THRESHOLD = 0.8

    # 交易时间
    TRADE_END_HOUR = 14
    TRADE_END_MINUTE = 50


# ==================== 趋势分析器 ====================
class TrendAnalyzer:
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"

    @staticmethod
    def calculate_ma(prices, period):
        if len(prices) < period:
            return None
        return np.mean(prices[-period:])

    @staticmethod
    def calculate_slope(prices, period=20):
        if len(prices) < period:
            return 0
        recent = prices[-period:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        avg_price = np.mean(recent)
        return slope / avg_price if avg_price > 0 else 0

    @staticmethod
    def check_ma_alignment(prices):
        """检查均线排列: 返回 (是否多头排列, 是否空头排列, 排列强度)"""
        ma20 = TrendAnalyzer.calculate_ma(prices, Config.TREND_MA_SHORT)
        ma60 = TrendAnalyzer.calculate_ma(prices, Config.TREND_MA_MID)
        ma120 = TrendAnalyzer.calculate_ma(prices, Config.TREND_MA_LONG)

        if ma20 is None or ma60 is None or ma120 is None:
            return False, False, 0

        current = prices[-1]
        # 多头排列: 价格 > MA20 > MA60 > MA120
        bull_align = current > ma20 > ma60 > ma120
        # 空头排列: 价格 < MA20 < MA60 < MA120
        bear_align = current < ma20 < ma60 < ma120

        # 计算排列强度 (均线间距)
        if bull_align:
            spread = (ma20 - ma120) / ma120
            strength = min(1.0, spread / 0.1)
        elif bear_align:
            spread = (ma120 - ma20) / ma120
            strength = min(1.0, spread / 0.1)
        else:
            strength = 0

        return bull_align, bear_align, strength

    @staticmethod
    def check_trend_consistency(prices):
        """检查多周期趋势一致性"""
        slope_short = TrendAnalyzer.calculate_slope(prices, 10)
        slope_mid = TrendAnalyzer.calculate_slope(prices, 20)
        slope_long = TrendAnalyzer.calculate_slope(prices, 60)

        # 三周期同向为一致
        all_up = slope_short > 0 and slope_mid > 0 and slope_long > 0
        all_down = slope_short < 0 and slope_mid < 0 and slope_long < 0

        if all_up:
            return 1.0  # 一致看涨
        elif all_down:
            return -1.0  # 一致看跌
        else:
            return 0  # 不一致

    @staticmethod
    def analyze(close_prices):
        """分析趋势状态 - 增强版"""
        if len(close_prices) < Config.TREND_MA_LONG:
            return TrendAnalyzer.SIDEWAYS, 0.5, 0

        current_price = close_prices[-1]
        ma20 = TrendAnalyzer.calculate_ma(close_prices, Config.TREND_MA_SHORT)
        ma60 = TrendAnalyzer.calculate_ma(close_prices, Config.TREND_MA_MID)

        # 计算斜率
        slope = TrendAnalyzer.calculate_slope(close_prices, 20)

        # 检查均线排列
        bull_align, bear_align, align_strength = TrendAnalyzer.check_ma_alignment(close_prices)

        # 检查趋势一致性
        consistency = TrendAnalyzer.check_trend_consistency(close_prices)

        # 综合判断趋势
        if bull_align and consistency > 0:
            # 多头排列 + 多周期一致 = 强牛市
            strength = min(1.0, (align_strength + abs(consistency)) / 2 + slope / 0.01)
            return TrendAnalyzer.BULL, strength, consistency
        elif bear_align and consistency < 0:
            # 空头排列 + 多周期一致 = 强熊市
            strength = min(1.0, (align_strength + abs(consistency)) / 2 + abs(slope) / 0.01)
            return TrendAnalyzer.BEAR, strength, consistency
        elif current_price > ma60 and slope > Config.TREND_SLOPE_THRESHOLD:
            # 价格在MA60上方且斜率为正 = 弱牛市
            strength = min(1.0, slope / 0.01) * 0.7
            return TrendAnalyzer.BULL, strength, consistency
        elif current_price < ma60 and slope < -Config.TREND_SLOPE_THRESHOLD:
            # 价格在MA60下方且斜率为负 = 弱熊市
            strength = min(1.0, abs(slope) / 0.01) * 0.7
            return TrendAnalyzer.BEAR, strength, consistency
        else:
            strength = 1.0 - min(1.0, abs(slope) / 0.005)
            return TrendAnalyzer.SIDEWAYS, strength, consistency


# ==================== 量价分析器 ====================
class VolumePriceAnalyzer:
    # 简化为4种核心状态 (增加容错度)
    HEALTHY_UP = "HEALTHY_UP"      # 健康上涨: 放量+动量向上
    WEAK_UP = "WEAK_UP"            # 弱势上涨/横盘: 缩量或弱动量
    SHRINK_DOWN = "SHRINK_DOWN"    # 普通下跌: 缩量+负动量
    PANIC_DOWN = "PANIC_DOWN"      # 危险信号: 放量下跌或出货

    @staticmethod
    def analyze(close_prices, volumes):
        """分析量价状态 - 简化版 (4种状态+双周期动量+容差判断)"""
        min_len = max(Config.VOLUME_MA_PERIOD, Config.PRICE_MOMENTUM_LONG + 1)
        if len(close_prices) < min_len:
            return VolumePriceAnalyzer.SHRINK_DOWN, 1.0, 0, 0

        # 量比计算
        vol_ma = np.mean(volumes[-Config.VOLUME_MA_PERIOD:])
        current_vol = volumes[-1]
        vol_ratio = current_vol / vol_ma if vol_ma > 0 else 1.0

        # 双周期动量验证
        mom_short = (close_prices[-1] - close_prices[-Config.PRICE_MOMENTUM_SHORT]) / close_prices[-Config.PRICE_MOMENTUM_SHORT]
        mom_long = (close_prices[-1] - close_prices[-Config.PRICE_MOMENTUM_LONG]) / close_prices[-Config.PRICE_MOMENTUM_LONG]

        # 简化判断逻辑 (增加容差)
        vol_high = vol_ratio > Config.VOLUME_RATIO_HIGH      # >1.2 放量
        vol_very_high = vol_ratio > Config.VOLUME_RATIO_VERY_HIGH  # >1.5 明显放量

        # 动量判断增加容差带 (±0.5%内视为中性)
        mom_tolerance = 0.005
        mom_up = mom_short > mom_tolerance and mom_long > -mom_tolerance  # 双动量偏正
        mom_down = mom_short < -mom_tolerance and mom_long < mom_tolerance  # 双动量偏负
        mom_weak = abs(mom_short) <= Config.MOMENTUM_WEAK  # 弱动量 ±1%

        # 4种状态判断
        # 1. PANIC_DOWN: 放量下跌 或 放量滞涨(出货)
        if vol_very_high and (mom_down or mom_weak):
            return VolumePriceAnalyzer.PANIC_DOWN, vol_ratio, mom_short, mom_long
        if vol_high and mom_short < -Config.MOMENTUM_STRONG:
            return VolumePriceAnalyzer.PANIC_DOWN, vol_ratio, mom_short, mom_long

        # 2. HEALTHY_UP: 放量上涨
        if vol_high and mom_up:
            return VolumePriceAnalyzer.HEALTHY_UP, vol_ratio, mom_short, mom_long

        # 3. SHRINK_DOWN: 缩量下跌
        if mom_down and not vol_high:
            return VolumePriceAnalyzer.SHRINK_DOWN, vol_ratio, mom_short, mom_long

        # 4. WEAK_UP: 其他情况(缩量上涨/横盘整理/弱势)
        return VolumePriceAnalyzer.WEAK_UP, vol_ratio, mom_short, mom_long


# ==================== 信号分析器 ====================
class SignalAnalyzer:
    # 信号类型
    DOUBLE_GOLD = "DOUBLE_GOLD"      # 双金叉
    DOUBLE_DEAD = "DOUBLE_DEAD"      # 双死叉
    TOP_DIVERGE = "TOP_DIVERGE"      # 顶背离
    BOTTOM_DIVERGE = "BOTTOM_DIVERGE"  # 底背离
    CONFLICT_BEAR = "CONFLICT_BEARISH"
    CONFLICT_BULL = "CONFLICT_BULLISH"
    MACD_GOLD = "MACD_GOLD"          # 单MACD金叉
    MACD_DEAD = "MACD_DEAD"          # 单MACD死叉
    NEUTRAL = "NEUTRAL"

    # 信号强度等级
    VERY_STRONG = "VERY_STRONG"
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"

    @staticmethod
    def calculate_macd(prices):
        if len(prices) < Config.MACD_SLOW + Config.MACD_SIGNAL:
            return None, None, None

        ema_fast = SignalAnalyzer._ema(prices, Config.MACD_FAST)
        ema_slow = SignalAnalyzer._ema(prices, Config.MACD_SLOW)
        dif = ema_fast - ema_slow

        dea_list = [dif[0]]
        k = 2 / (Config.MACD_SIGNAL + 1)
        for i in range(1, len(dif)):
            dea_list.append(dif[i] * k + dea_list[-1] * (1 - k))
        dea = np.array(dea_list)
        histogram = 2 * (dif - dea)

        return dif, dea, histogram

    @staticmethod
    def _ema(prices, period):
        result = [prices[0]]
        k = 2 / (period + 1)
        for i in range(1, len(prices)):
            result.append(prices[i] * k + result[-1] * (1 - k))
        return np.array(result)

    @staticmethod
    def calculate_ma_cross(prices):
        if len(prices) < Config.MA_LONG:
            return None, None
        ma_short = np.convolve(prices, np.ones(Config.MA_SHORT)/Config.MA_SHORT, 'valid')
        ma_long = np.convolve(prices, np.ones(Config.MA_LONG)/Config.MA_LONG, 'valid')
        min_len = min(len(ma_short), len(ma_long))
        return ma_short[-min_len:], ma_long[-min_len:]

    @staticmethod
    def detect_divergence(prices, dif, lookback=30):
        """检测背离: 价格与MACD的背离"""
        if len(prices) < lookback or len(dif) < lookback:
            return None

        prices_recent = prices[-lookback:]
        dif_recent = dif[-lookback:]

        # 找局部高点和低点
        price_highs = []
        price_lows = []
        dif_highs = []
        dif_lows = []

        for i in range(2, lookback - 2):
            # 局部高点
            if prices_recent[i] > prices_recent[i-1] and prices_recent[i] > prices_recent[i-2] and \
               prices_recent[i] > prices_recent[i+1] and prices_recent[i] > prices_recent[i+2]:
                price_highs.append((i, prices_recent[i]))
                dif_highs.append((i, dif_recent[i]))
            # 局部低点
            if prices_recent[i] < prices_recent[i-1] and prices_recent[i] < prices_recent[i-2] and \
               prices_recent[i] < prices_recent[i+1] and prices_recent[i] < prices_recent[i+2]:
                price_lows.append((i, prices_recent[i]))
                dif_lows.append((i, dif_recent[i]))

        # 顶背离: 价格创新高，MACD未创新高
        if len(price_highs) >= 2 and len(dif_highs) >= 2:
            if price_highs[-1][1] > price_highs[-2][1] and dif_highs[-1][1] < dif_highs[-2][1]:
                return SignalAnalyzer.TOP_DIVERGE

        # 底背离: 价格创新低，MACD未创新低
        if len(price_lows) >= 2 and len(dif_lows) >= 2:
            if price_lows[-1][1] < price_lows[-2][1] and dif_lows[-1][1] > dif_lows[-2][1]:
                return SignalAnalyzer.BOTTOM_DIVERGE

        return None

    @staticmethod
    def check_cross_in_window(series1, series2, window=3):
        """检查近N天内是否发生交叉"""
        if len(series1) < window + 1 or len(series2) < window + 1:
            return 0

        for i in range(1, window + 1):
            idx = -i
            prev_idx = idx - 1
            # 金叉
            if series1[prev_idx] <= series2[prev_idx] and series1[idx] > series2[idx]:
                return 1
            # 死叉
            if series1[prev_idx] >= series2[prev_idx] and series1[idx] < series2[idx]:
                return -1
        return 0

    @staticmethod
    def get_signal_strength(dif, hist, macd_cross):
        """计算信号强度等级"""
        # 基于零轴位置和柱状图变化
        above_zero = dif[-1] > 0
        hist_expanding = len(hist) >= 2 and abs(hist[-1]) > abs(hist[-2])

        if macd_cross == 1:  # 金叉
            if above_zero and hist_expanding:
                return SignalAnalyzer.VERY_STRONG, 1.0
            elif above_zero:
                return SignalAnalyzer.STRONG, 0.8
            elif hist_expanding:
                return SignalAnalyzer.MODERATE, 0.6
            else:
                return SignalAnalyzer.WEAK, 0.4
        elif macd_cross == -1:  # 死叉
            if not above_zero and hist_expanding:
                return SignalAnalyzer.VERY_STRONG, 1.0
            elif not above_zero:
                return SignalAnalyzer.STRONG, 0.8
            elif hist_expanding:
                return SignalAnalyzer.MODERATE, 0.6
            else:
                return SignalAnalyzer.WEAK, 0.4
        return SignalAnalyzer.WEAK, 0.3

    @staticmethod
    def analyze(close_prices):
        """分析技术信号 - 增强版"""
        dif, dea, hist = SignalAnalyzer.calculate_macd(close_prices)
        ma_short, ma_long = SignalAnalyzer.calculate_ma_cross(close_prices)

        if dif is None or ma_short is None:
            return SignalAnalyzer.NEUTRAL, 0, SignalAnalyzer.WEAK

        # 检测背离 (优先级最高)
        divergence = SignalAnalyzer.detect_divergence(close_prices, dif, Config.DIVERGENCE_LOOKBACK)
        if divergence == SignalAnalyzer.TOP_DIVERGE:
            return SignalAnalyzer.TOP_DIVERGE, 0.9, SignalAnalyzer.STRONG
        elif divergence == SignalAnalyzer.BOTTOM_DIVERGE:
            return SignalAnalyzer.BOTTOM_DIVERGE, 0.9, SignalAnalyzer.STRONG

        # 检测近3天内的交叉 (扩展检测窗口)
        macd_cross = SignalAnalyzer.check_cross_in_window(dif, dea, window=3)
        ma_cross = SignalAnalyzer.check_cross_in_window(ma_short, ma_long, window=3)

        # 获取信号强度
        strength_level, strength_score = SignalAnalyzer.get_signal_strength(dif, hist, macd_cross)

        # 判断信号类型
        if macd_cross == 1 and ma_cross == 1:
            return SignalAnalyzer.DOUBLE_GOLD, strength_score, strength_level
        elif macd_cross == -1 and ma_cross == -1:
            return SignalAnalyzer.DOUBLE_DEAD, strength_score, strength_level
        elif macd_cross == 1 and ma_cross == -1:
            return SignalAnalyzer.CONFLICT_BULL, 0.4, SignalAnalyzer.WEAK
        elif macd_cross == -1 and ma_cross == 1:
            return SignalAnalyzer.CONFLICT_BEAR, 0.4, SignalAnalyzer.WEAK
        elif macd_cross == 1:
            return SignalAnalyzer.MACD_GOLD, strength_score * 0.7, strength_level
        elif macd_cross == -1:
            return SignalAnalyzer.MACD_DEAD, strength_score * 0.7, strength_level
        else:
            return SignalAnalyzer.NEUTRAL, 0, SignalAnalyzer.WEAK


# ==================== 仓位管理器 ====================
class PositionManager:
    # 仓位矩阵: 3趋势 x 4量价 = 12种组合
    POSITION_MATRIX = {
        # 牛市仓位 (进攻为主)
        (TrendAnalyzer.BULL, VolumePriceAnalyzer.HEALTHY_UP): 0.90,   # 最佳状态
        (TrendAnalyzer.BULL, VolumePriceAnalyzer.WEAK_UP): 0.75,      # 观望上涨
        (TrendAnalyzer.BULL, VolumePriceAnalyzer.SHRINK_DOWN): 0.55,  # 回调缩量
        (TrendAnalyzer.BULL, VolumePriceAnalyzer.PANIC_DOWN): 0.35,   # 危险信号

        # 熊市仓位 (严格防守)
        (TrendAnalyzer.BEAR, VolumePriceAnalyzer.HEALTHY_UP): 0.35,   # 反弹
        (TrendAnalyzer.BEAR, VolumePriceAnalyzer.WEAK_UP): 0.20,      # 弱反弹
        (TrendAnalyzer.BEAR, VolumePriceAnalyzer.SHRINK_DOWN): 0.10,  # 阴跌
        (TrendAnalyzer.BEAR, VolumePriceAnalyzer.PANIC_DOWN): 0.0,    # 恐慌

        # 震荡市仓位 (灵活应对)
        (TrendAnalyzer.SIDEWAYS, VolumePriceAnalyzer.HEALTHY_UP): 0.65,  # 震荡放量
        (TrendAnalyzer.SIDEWAYS, VolumePriceAnalyzer.WEAK_UP): 0.50,     # 震荡弱势
        (TrendAnalyzer.SIDEWAYS, VolumePriceAnalyzer.SHRINK_DOWN): 0.35, # 震荡缩量
        (TrendAnalyzer.SIDEWAYS, VolumePriceAnalyzer.PANIC_DOWN): 0.15,  # 震荡恐慌
    }

    # 信号系数
    SIGNAL_COEFFICIENT = {
        SignalAnalyzer.DOUBLE_GOLD: 1.15,     # 双金叉强买
        SignalAnalyzer.DOUBLE_DEAD: 0.3,      # 双死叉强卖
        SignalAnalyzer.TOP_DIVERGE: 0.5,      # 顶背离减仓
        SignalAnalyzer.BOTTOM_DIVERGE: 1.1,   # 底背离加仓
        SignalAnalyzer.MACD_GOLD: 1.08,       # 单金叉轻微加
        SignalAnalyzer.MACD_DEAD: 0.7,        # 单死叉轻微减
        SignalAnalyzer.CONFLICT_BEAR: 0.6,    # 冲突偏空
        SignalAnalyzer.CONFLICT_BULL: 0.9,    # 冲突偏多
        SignalAnalyzer.NEUTRAL: 1.0,          # 无信号
    }

    # 信号强度等级系数
    STRENGTH_COEFFICIENT = {
        SignalAnalyzer.VERY_STRONG: 1.1,      # 非常强信号
        SignalAnalyzer.STRONG: 1.05,          # 强信号
        SignalAnalyzer.MODERATE: 1.0,         # 中等信号
        SignalAnalyzer.WEAK: 0.9,             # 弱信号
    }

    @staticmethod
    def calculate_target_position(trend, vp_state, signal, signal_strength=None, trend_consistency=0):
        """计算目标仓位 - 含信号强度等级系数"""
        base_position = PositionManager.POSITION_MATRIX.get((trend, vp_state), 0.3)
        signal_coef = PositionManager.SIGNAL_COEFFICIENT.get(signal, 1.0)

        # 信号强度等级系数 - 调整信号系数的影响力
        strength_coef = PositionManager.STRENGTH_COEFFICIENT.get(signal_strength, 1.0)

        # 对信号系数应用强度调整 (偏离1.0的部分按强度系数放大或缩小)
        # 例如: 信号系数1.15, 强度系数1.15 -> 1 + (1.15-1)*1.15 = 1.1725
        # 例如: 信号系数0.7, 强度系数0.85 -> 1 + (0.7-1)*0.85 = 0.745 (弱化减仓信号)
        adjusted_signal_coef = 1.0 + (signal_coef - 1.0) * strength_coef

        # 趋势一致性调整 (简化为单一系数)
        consistency_coef = 1.0
        if trend_consistency > 0:
            consistency_coef = 1.08  # 多周期一致看涨
        elif trend_consistency < 0:
            consistency_coef = 0.92  # 多周期一致看跌

        target = base_position * adjusted_signal_coef * consistency_coef
        return min(1.0, max(0.0, target))

    @staticmethod
    def should_adjust(current_ratio, target_ratio, threshold=0.05):
        return abs(current_ratio - target_ratio) > threshold


# ==================== PTrade 策略主体 ====================

def initialize(context):
    """初始化函数"""
    context.stock = Config.STOCK
    set_benchmark(Config.BENCHMARK)
    set_universe([context.stock])

    # ========== 关键修复：PTrade正确的佣金和滑点设置 ==========
    # PTrade不支持PerOrder类，使用set_commission和set_slippage
    set_commission(commission_ratio=0.0003, min_commission=5.0)
    set_slippage(slippage=0.002)

    # 状态变量
    context.trend = TrendAnalyzer.SIDEWAYS
    context.trend_strength = 0.5
    context.trend_consistency = 0      # 趋势一致性
    context.vp_state = VolumePriceAnalyzer.SHRINK_DOWN
    context.signal = SignalAnalyzer.NEUTRAL
    context.signal_strength = SignalAnalyzer.WEAK  # 信号强度等级
    context.target_position = 0.0

    # 风控变量
    context.peak_value = 0                 # 账户历史最高净值（重置后更新）
    context.stop_loss_triggered = False
    context.cooldown_end_date = None

    # T+0追踪 - 关键：PTrade没有closeable_amount，需手动追踪
    context.t0_buy_today = 0           # 当日买入量
    context.t0_sold_today = False
    context.t0_sell_price = 0
    context.t0_sell_time = None
    context.t0_sell_amount = 0         # 记录T+0卖出数量
    context.t0_rebuy_done = False      # T+0回补已完成，当日跳过收盘调仓

    # 昨收价缓存
    context.prev_close = 0

    log.info("=" * 60)
    log.info("StrategyThr v1.3 初始化完成")
    log.info("标的: %s | 基准: %s" % (context.stock, Config.BENCHMARK))
    log.info("趋势回溯: %d天 | 信号回溯: %d天" % (Config.TREND_LOOKBACK, Config.SIGNAL_LOOKBACK))
    log.info("止损: %.0f%% | 账户回撤: %.0f%%清仓 | 止损冷却: %d天 | 回撤冷却: %d天" %
            (Config.STOP_LOSS_THRESHOLD*100, Config.MAX_DRAWDOWN*100,
             Config.COOLDOWN_DAYS, Config.DRAWDOWN_COOLDOWN_DAYS))
    log.info("T+0: 涨%.0f%%卖出100%%, 回落%.0f%%补足目标仓位" %
            (Config.T0_SELL_THRESHOLD*100, Config.T0_REBUY_THRESHOLD*100))
    log.info("=" * 60)


def before_trading_start(context, data):
    """盘前分析"""
    # 重置T+0状态
    context.t0_buy_today = 0
    context.t0_sold_today = False
    context.t0_sell_price = 0
    context.t0_sell_time = None
    context.t0_sell_amount = 0
    context.t0_rebuy_done = False

    # 获取历史数据
    try:
        hist = get_history(Config.TREND_LOOKBACK, '1d', ['close', 'volume'],
                          context.stock, fq='pre', include=False)
        if hist is None or len(hist) < 60:
            log.warning("历史数据不足")
            return

        close_prices = hist['close'].values
        volumes = hist['volume'].values

        # 缓存昨收价
        context.prev_close = close_prices[-1]

        # 趋势分析 (返回: 趋势, 强度, 一致性)
        context.trend, context.trend_strength, context.trend_consistency = TrendAnalyzer.analyze(close_prices)

        # 量价分析 (返回: 状态, 量比, 短动量, 长动量)
        context.vp_state, vol_ratio, mom_short, mom_long = VolumePriceAnalyzer.analyze(close_prices, volumes)

        # 信号分析 (返回: 信号, 强度分数, 强度等级)
        context.signal, signal_score, context.signal_strength = SignalAnalyzer.analyze(close_prices)

        # 计算目标仓位
        context.target_position = PositionManager.calculate_target_position(
            context.trend, context.vp_state, context.signal,
            context.signal_strength, context.trend_consistency)

        # 日志
        log.info("-" * 40)
        log.info("日期: %s" % str(context.blotter.current_dt.date()))
        log.info("趋势: %s (强度: %.2f, 一致性: %.1f)" % (context.trend, context.trend_strength, context.trend_consistency))
        log.info("量价状态: %s (量比: %.2f, 短动量: %.2f%%, 长动量: %.2f%%)" %
                (context.vp_state, vol_ratio, mom_short*100, mom_long*100))
        log.info("技术信号: %s (强度: %s, 分数: %.2f)" % (context.signal, context.signal_strength, signal_score))
        log.info("目标仓位: %.1f%%" % (context.target_position * 100))

        # 冷却期提示
        if context.cooldown_end_date:
            current_date = context.blotter.current_dt.date()
            if current_date < context.cooldown_end_date:
                remaining = (context.cooldown_end_date - current_date).days
                log.warning("冷却期中，剩余%d天，所有交易暂停" % remaining)

        log.info("-" * 40)

    except Exception as e:
        log.error("[before_trading_start异常] %s" % str(e))


def handle_data(context, data):
    """盘中处理 - 每分钟执行"""
    try:
        security = context.stock

        # 获取当前价格 - PTrade BarData访问方式
        if security not in data:
            return
        bar = data[security]
        current_price = bar.close

        if current_price <= 0:
            return

        # ===== 冷却期检查 - 所有交易暂停 =====
        current_date = context.blotter.current_dt.date()
        if context.cooldown_end_date and current_date < context.cooldown_end_date:
            return  # 冷却期内不执行任何交易

        # 更新账户峰值（历史最高净值）
        total_value = context.portfolio.total_value
        if total_value > context.peak_value:
            context.peak_value = total_value

        # 获取当前持仓
        position = context.portfolio.positions.get(security)
        current_shares = position.amount if position else 0

        # ===== 关键修复：计算可卖数量 =====
        # PTrade没有closeable_amount属性，需要手动追踪
        # 可卖数量 = 总持仓 - 当日买入量
        closeable_shares = max(0, current_shares - context.t0_buy_today)

        current_value = current_shares * current_price
        position_ratio = current_value / total_value if total_value > 0 else 0

        # ===== 风控检查 =====
        if position and current_shares > 0:
            cost_basis = position.cost_basis if hasattr(position, 'cost_basis') else position.avg_cost
            loss_ratio = (cost_basis - current_price) / cost_basis

            # 止损触发条件: 成本亏损>=12%
            if loss_ratio >= Config.STOP_LOSS_THRESHOLD:
                log.warning("[止损触发] 成本亏损: %.2f%%" % (loss_ratio * 100))
                if closeable_shares >= 100:
                    order(security, -closeable_shares)
                    log.info("[止损卖出] 数量: %d" % closeable_shares)
                # 固定冷却期3天
                context.cooldown_end_date = current_date + timedelta(days=Config.COOLDOWN_DAYS)
                log.warning("[进入冷却期] 冷却%d天，暂停至 %s" %
                           (Config.COOLDOWN_DAYS, str(context.cooldown_end_date)))
                return

        # 账户净值回撤检查 - 从历史高点回撤>=15%则清仓
        drawdown = (context.peak_value - total_value) / context.peak_value if context.peak_value > 0 else 0
        if drawdown >= Config.MAX_DRAWDOWN:
            log.warning("[账户回撤清仓] 从峰值%.2f回撤%.2f%%" %
                       (context.peak_value, drawdown * 100))
            if closeable_shares >= 100:
                order(security, -closeable_shares)
                log.info("[回撤清仓] 卖出数量: %d" % closeable_shares)
            # 回撤触发后，以当前账户价值作为新峰值基准
            context.peak_value = total_value
            log.info("[峰值重置] 新峰值基准: %.2f" % total_value)
            # 固定冷却期1天
            context.cooldown_end_date = current_date + timedelta(days=Config.DRAWDOWN_COOLDOWN_DAYS)
            log.warning("[进入冷却期] 冷却%d天，暂停至 %s" %
                       (Config.DRAWDOWN_COOLDOWN_DAYS, str(context.cooldown_end_date)))
            return

        # ===== T+0交易逻辑 =====
        current_time = context.blotter.current_dt
        prev_close = context.prev_close if context.prev_close > 0 else current_price

        # T+0卖出检查 - 100%卖出可卖仓位
        if not context.t0_sold_today and closeable_shares >= 100:
            if position_ratio >= Config.T0_POSITION_THRESHOLD:
                gain = (current_price - prev_close) / prev_close
                if gain >= Config.T0_SELL_THRESHOLD:
                    # 100%卖出所有可卖仓位
                    sell_amount = int(closeable_shares / 100) * 100
                    if sell_amount >= 100:
                        order(security, -sell_amount)
                        context.t0_sold_today = True
                        context.t0_sell_price = current_price
                        context.t0_sell_time = current_time
                        context.t0_sell_amount = sell_amount  # 记录卖出数量
                        log.info("[T+0卖出] 数量: %d 价格: %.2f 涨幅: %.2f%% (100%%卖出)" %
                                (sell_amount, current_price, gain * 100))

        # T+0回补检查 - 补足目标仓位
        if context.t0_sold_today and context.t0_sell_price > 0:
            minutes_since_sell = (current_time - context.t0_sell_time).seconds // 60
            if minutes_since_sell >= Config.T0_COOLDOWN_MINUTES:
                drop = (context.t0_sell_price - current_price) / context.t0_sell_price
                if drop >= Config.T0_REBUY_THRESHOLD:
                    # 计算目标仓位所需股数
                    target_ratio = context.target_position
                    target_value = total_value * target_ratio
                    target_shares = int(target_value / current_price / 100) * 100

                    # 买入量 = 目标股数 - 当前持仓
                    buy_amount = target_shares - current_shares
                    buy_amount = int(buy_amount / 100) * 100

                    # 确保不超过可用现金
                    if buy_amount > 0:
                        cash = context.portfolio.cash
                        max_buy = int(cash * 0.98 / current_price / 100) * 100
                        buy_amount = min(buy_amount, max_buy)

                        if buy_amount >= 100:
                            order(security, buy_amount)
                            context.t0_buy_today += buy_amount  # 追踪当日买入
                            context.t0_rebuy_done = True  # 标记回补完成，跳过收盘调仓
                            new_shares = current_shares + buy_amount
                            new_ratio = new_shares * current_price / total_value * 100
                            log.info("[T+0回补] 数量: %d 价格: %.2f 回落: %.2f%% 目标仓位: %.1f%% 回补后: %.1f%%" %
                                    (buy_amount, current_price, drop * 100, target_ratio * 100, new_ratio))
                        context.t0_sold_today = False
                    else:
                        # 当前仓位已达到或超过目标，无需回补
                        log.info("[T+0回补跳过] 当前仓位已达目标，无需买入")
                        context.t0_sold_today = False
                        context.t0_rebuy_done = True

        # ===== 收盘前调仓 =====
        if current_time.hour == Config.TRADE_END_HOUR and current_time.minute >= Config.TRADE_END_MINUTE:
            # T+0回补已完成时，跳过收盘调仓
            if context.t0_rebuy_done:
                return

            target_ratio = context.target_position

            if PositionManager.should_adjust(position_ratio, target_ratio, Config.POSITION_CHANGE_THRESHOLD):
                target_value = total_value * target_ratio
                target_shares = int(target_value / current_price / 100) * 100
                diff_shares = target_shares - current_shares

                if diff_shares > 0:
                    # 买入
                    cash = context.portfolio.cash
                    max_buy = int(cash * 0.98 / current_price / 100) * 100
                    buy_amount = min(diff_shares, max_buy)
                    if buy_amount >= 100:
                        order(security, buy_amount)
                        context.t0_buy_today += buy_amount  # 追踪当日买入
                        log.info("[买入] 数量: %d 价格: %.2f 目标仓位: %.1f%%" %
                                (buy_amount, current_price, target_ratio * 100))

                elif diff_shares < 0:
                    # 卖出 - 只能卖可卖部分
                    sell_amount = min(abs(diff_shares), closeable_shares)
                    sell_amount = int(sell_amount / 100) * 100
                    if sell_amount >= 100:
                        order(security, -sell_amount)
                        log.info("[卖出] 数量: %d 价格: %.2f 目标仓位: %.1f%%" %
                                (sell_amount, current_price, target_ratio * 100))

    except Exception as e:
        log.error("[handle_data异常] %s" % str(e))


def after_trading_end(context, data):
    """盘后处理"""
    security = context.stock
    position = context.portfolio.positions.get(security)
    current_shares = position.amount if position else 0
    cost_basis = 0
    if position:
        if hasattr(position, 'cost_basis'):
            cost_basis = position.cost_basis
        elif hasattr(position, 'avg_cost'):
            cost_basis = position.avg_cost

    total_value = context.portfolio.total_value

    log.info("[收盘] 持仓: %d股 成本: %.2f 总资产: %.2f" %
            (current_shares, cost_basis, total_value))

    # 冷却期状态
    if context.cooldown_end_date:
        current_date = context.blotter.current_dt.date()
        if current_date < context.cooldown_end_date:
            remaining = (context.cooldown_end_date - current_date).days
            log.warning("[冷却期] 剩余%d天，明日所有交易继续暂停" % remaining)
        else:
            log.info("[冷却期结束] 明日恢复正常交易")
            context.cooldown_end_date = None