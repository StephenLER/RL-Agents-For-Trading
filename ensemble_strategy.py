import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# 解决 OMP 报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import PPO, A2C, DDPG
from env_stocktrading import StockTradingEnv

# --- 配置参数 ---
DATA_PATH = "./data/final_rl_data.csv"
TRAINED_MODEL_DIR = "./trained_models"
RESULTS_DIR = "./results"

# 回测区间 (2024-2025)
TRADE_START_DATE = "2024-01-01"
TRADE_END_DATE = "2025-12-19"

# 集成参数 (论文逻辑: Quarterly)
REBALANCE_WINDOW = 63  # 约 3 个月 (一季度)
VALIDATION_WINDOW = 63 # 回看过去 3 个月来决定选谁

def data_split(df, start, end):
    data = df[(df.date >= start) & (df.date < end)]
    data = data.sort_values(["date", "tic"], ignore_index=True)
    data.index = data.date.factorize()[0]
    return data

def calculate_sharpe(df_account):
    """计算夏普比率 [cite: 264-265]"""
    df = df_account.copy()
    df['daily_return'] = df['account_value'].pct_change(1)
    if df['daily_return'].std() == 0: return 0
    
    annual_return = df['daily_return'].mean() * 252
    annual_volatility = df['daily_return'].std() * np.sqrt(252)
    risk_free_rate = 0.03
    return (annual_return - risk_free_rate) / annual_volatility

def get_validation_performance(agent_name, model, df_val, stock_dim):
    """
    在验证数据上跑一遍模型，计算夏普比率
    """
    if len(df_val) == 0: return -999
    
    # 构建临时环境
    env_kwargs = {
        "df": df_val,
        "stock_dim": stock_dim,
        "hmax": 100,
        "initial_amount": 1000000, 
        "transaction_cost_pct": 0.0003,
        "stamp_duty_pct": 0.0005,
        "reward_scaling": 1e-4,
        "turbulence_threshold": None
    }
    env = StockTradingEnv(**env_kwargs)
    
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        
    df_res = env.save_asset_memory()
    sharpe = calculate_sharpe(df_res)
    return sharpe

def run_ensemble_strategy():
    print("====== 开始运行集成策略 (Ensemble Strategy) ======")
    
    # 1. 准备数据
    df = pd.read_csv(DATA_PATH)
    trade_df = data_split(df, TRADE_START_DATE, TRADE_END_DATE)
    stock_dim = len(trade_df.tic.unique())
    unique_trade_date = trade_df.date.unique()
    
    print(f"回测时间: {TRADE_START_DATE} ~ {TRADE_END_DATE}")
    print(f"总交易日: {len(unique_trade_date)} 天")
    print(f"调仓周期: 每 {REBALANCE_WINDOW} 天")

    # 2. 加载三个训练好的模型 [cite: 50]
    print("正在加载模型...")
    models = {
        "PPO": PPO.load(f"{TRAINED_MODEL_DIR}/agent_ppo"),
        "A2C": A2C.load(f"{TRAINED_MODEL_DIR}/agent_a2c"),
        "DDPG": DDPG.load(f"{TRAINED_MODEL_DIR}/agent_ddpg")
    }

    # 3. 初始化集成环境 (主环境)
    env_kwargs = {
        "df": trade_df,
        "stock_dim": stock_dim,
        "hmax": 100,
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.0003,
        "stamp_duty_pct": 0.0005,
        "reward_scaling": 1e-4,
        "turbulence_threshold": None
    }
    env_ensemble = StockTradingEnv(**env_kwargs)
    env_ensemble.reset()

    # 4. 滚动回测循环
    # 初始默认使用 PPO (或者你可以指定 DDPG)
    current_agent_name = "PPO" 
    current_model = models[current_agent_name]
    
    # 记录模型切换历史
    model_history = [] 

    obs_ensemble, _ = env_ensemble.reset()
    
    for i in range(len(unique_trade_date)):
        # --- A. 检查是否需要换人 (每 3 个月) ---
        # 我们必须确保有足够的历史数据来验证 (i >= VALIDATION_WINDOW)
        if i > 0 and i % REBALANCE_WINDOW == 0:
            print(f"\n[Day {i}] 触发季度调仓检查...")
            
            # 获取过去 3 个月的数据片段 (用于验证)
            # 注意: 这里需要从原始大表中切片，为了简化，我们直接从 trade_df 切
            start_idx = i - VALIDATION_WINDOW
            end_idx = i
            
            # 这里的切片逻辑需要小心，因为 data_split 重置了 index
            # 我们需要找出对应的日期，重新切分 DataFrame 传给环境
            val_start_date = unique_trade_date[start_idx]
            val_end_date = unique_trade_date[end_idx]
            
            # 从原始 df 中切出这段验证数据 (包含完整的股票数据)
            df_val = data_split(df, val_start_date, val_end_date)
            
            print(f"  验证区间: {val_start_date} ~ {val_end_date}")
            
            # 评估三个模型在过去 3 个月的表现
            sharpe_scores = {}
            for name, model in models.items():
                sharpe = get_validation_performance(name, model, df_val, stock_dim)
                sharpe_scores[name] = sharpe
                print(f"  - {name} 过去3个月 Sharpe: {sharpe:.4f}")
            
            # [cite: 263] pick the best performing agent
            best_agent_name = max(sharpe_scores, key=sharpe_scores.get)
            
            if best_agent_name != current_agent_name:
                print(f"  🔄 切换模型: {current_agent_name} -> {best_agent_name}")
                current_agent_name = best_agent_name
                current_model = models[current_agent_name]
            else:
                print(f"  ✅ 保持模型: {current_agent_name}")

        # 记录当前使用的模型
        model_history.append(current_agent_name)

        # --- B. 执行交易 ---
        # 使用当前选定的模型预测动作
        action, _ = current_model.predict(obs_ensemble, deterministic=True)
        obs_ensemble, rewards, done, _, _ = env_ensemble.step(action)
        
        if done:
            break

    # 5. 保存与画图
    df_result = env_ensemble.save_asset_memory()
    
    # 增加一列：显示每天用了哪个模型
    df_result['Agent'] = model_history[:len(df_result)]
    
    # 计算最终指标
    final_sharpe = calculate_sharpe(df_result)
    cum_return = (df_result.iloc[-1]['account_value'] - 1000000) / 1000000
    
    print("\n" + "="*30)
    print(f"集成策略回测完成！")
    print(f"累计收益: {cum_return*100:.2f}%")
    print(f"整体 Sharpe Ratio: {final_sharpe:.2f}")
    print("="*30)
    
    # 保存结果
    df_result.to_csv(f"{RESULTS_DIR}/ensemble_account_value.csv", index=False)
    
    # 画图
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(df_result['date']), df_result['account_value'], label='Ensemble Strategy', color='red')
    
    # 在图上标记模型切换点 (可选)
    # ...简单起见只画资金曲线
    
    plt.title(f"Ensemble Strategy Backtest (Sharpe: {final_sharpe:.2f})")
    plt.xlabel("Date")
    plt.ylabel("Account Value")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/ensemble_result.png")
    plt.show()

    # 打印模型使用统计
    from collections import Counter
    print("模型使用天数统计:", Counter(model_history))

if __name__ == "__main__":
    run_ensemble_strategy()