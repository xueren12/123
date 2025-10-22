import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv(r'c:\Users\64789\Desktop\123\data\start_20251001_end_20251014_tf_15min_inst_ETH\backtest_multi_indicator.csv')

# 转换时间戳
df['ts'] = pd.to_datetime(df['ts'])
df['date'] = df['ts'].dt.date

# 筛选10月12日的数据
oct12_data = df[df['date'] == pd.to_datetime('2025-10-12').date()].copy()

print('=== 10月12日ETH价格波动分析 ===')
print(f'数据点数: {len(oct12_data)}')
print(f'价格范围: {oct12_data["close"].min():.2f} - {oct12_data["close"].max():.2f} USDT')
print(f'最大单日波幅: {((oct12_data["close"].max() - oct12_data["close"].min()) / oct12_data["close"].min() * 100):.2f}%')

# 计算价格变化
oct12_data['price_change'] = oct12_data['close'].pct_change()
print(f'最大单期涨幅: {oct12_data["price_change"].max() * 100:.2f}%')
print(f'最大单期跌幅: {oct12_data["price_change"].min() * 100:.2f}%')

# 分析交易操作
trades_oct12 = oct12_data[oct12_data['signal'].isin(['BUY', 'SELL'])].copy()
print(f'\n=== 10月12日交易操作 ({len(trades_oct12)}笔) ===')

for _, trade in trades_oct12.iterrows():
    time_str = trade['ts'].strftime('%H:%M')
    price = trade['close']
    signal = trade['signal']
    position = trade['pos']
    equity = trade['equity']
    ret = trade['ret']
    print(f'{time_str} - {signal} @ {price:.2f}, 仓位: {position:.1f}, 收益: {ret:.4f}, 权益: {equity:.4f}')

# 分析权益变化和风险
print(f'\n=== 风险分析 ===')
print(f'当日起始权益: {oct12_data.iloc[0]["equity"]:.4f}')
print(f'当日结束权益: {oct12_data.iloc[-1]["equity"]:.4f}')
print(f'当日权益变化: {((oct12_data.iloc[-1]["equity"] / oct12_data.iloc[0]["equity"]) - 1) * 100:.2f}%')

# 检查是否有爆仓风险（权益大幅下降）
min_equity = oct12_data['equity'].min()
max_equity = oct12_data['equity'].max()
print(f'当日最低权益: {min_equity:.4f}')
print(f'当日最高权益: {max_equity:.4f}')
print(f'最大回撤: {((min_equity / max_equity) - 1) * 100:.2f}%')

# 分析仓位变化
print(f'\n=== 仓位管理 ===')
max_pos = oct12_data['pos'].max()
min_pos = oct12_data['pos'].min()
print(f'最大仓位: {max_pos:.1f}')
print(f'最小仓位: {min_pos:.1f}')

# 检查是否有强制平仓
if min_equity < 0.1:
    print('⚠️ 警告: 检测到爆仓风险!')
else:
    print('✅ 未检测到爆仓风险')

# 详细分析价格剧烈波动时段
print(f'\n=== 价格剧烈波动分析 ===')
# 找出价格变化超过1%的时段
large_moves = oct12_data[abs(oct12_data['price_change']) > 0.01].copy()
if len(large_moves) > 0:
    print('价格剧烈波动时段:')
    for _, move in large_moves.iterrows():
        time_str = move['ts'].strftime('%H:%M')
        change = move['price_change'] * 100
        price = move['close']
        signal = move['signal']
        pos = move['pos']
        equity = move['equity']
        print(f'  {time_str}: 价格变化 {change:+.2f}% @ {price:.2f}, 信号: {signal}, 仓位: {pos:.1f}, 权益: {equity:.4f}')
else:
    print('未发现单期超过1%的剧烈波动')