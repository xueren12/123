import pandas as pd
from pathlib import Path

# 配置
INITIAL_BALANCE = 1000  # USDT
BASE_DIR = Path(r"c:\Users\64789\Desktop\123\data\start_20250101_end_20251015_tf_15min_inst_ETH")
CSV_PATH = BASE_DIR / "backtest_multi_indicator.csv"

def main():
    """主函数，执行计算和输出"""
    try:
        df = pd.read_csv(CSV_PATH)
        if df.empty:
            print("错误：回测CSV文件为空。")
            return
        last_record = df.iloc[-1]

        # 提取最终数据
        final_equity_multiplier = float(last_record["equity"])
        final_pos_units = float(last_record["pos"])
        final_price = float(last_record["close"])

        # --- 计算 ---
        # 1. 最终账户总余额 (Total Equity)
        # 算法: 初始本金 * 最终权益乘数
        final_balance_usdt = INITIAL_BALANCE * final_equity_multiplier

        # 2. 最终仓位价值
        # 假设'pos'列代表资产单位（例如ETH数量）
        # 算法: 持有单位 * 最终价格
        position_value_usdt = final_pos_units * final_price

        # --- 输出结果 ---
        print("=== 策略回测账户状态评估 ===")
        print(f"初始本金: {INITIAL_BALANCE} USDT")
        print(f"回测周期: 2025-01-01 至 2025-10-15")
        print("-" * 35)

        print(f"账户最终总余额 (Equity): {final_balance_usdt:.2f} USDT")
        print(f"当前持有仓位价值 (Position Value): {position_value_usdt:.2f} USDT")
        print("-" * 35)

        print("详细信息:")
        print(f"- 最终权益乘数: {final_equity_multiplier:.4f}")
        print(f"- 最终收盘价 (ETH): {final_price:.2f} USDT")
        if final_pos_units > 0:
            print(f"- 当前仓位: {final_pos_units:.2f} 单位 (ETH)")
            print(f"\n注意: \'账户总余额\'是您账户的总价值，已包含当前持仓的未实现盈亏。")
        else:
            print("- 当前仓位: 空仓")

    except FileNotFoundError:
        print(f"错误: 文件未找到 {CSV_PATH}")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

if __name__ == "__main__":
    main()