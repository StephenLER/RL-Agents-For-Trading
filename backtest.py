import os
# --- 修复 OMP 报错的核心代码 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO, A2C, DDPG
from env_stocktrading import StockTradingEnv

# --- 配置参数 ---
DATA_PATH = "./data/final_rl_data.csv"
TRAINED_MODEL_DIR = "./trained_models"
RESULTS_DIR = "./results"

# 测试集日期 (必须与 train.py 中的 TRADE 日期一致)
TRADE_START_DATE = "2024-01-01"
TRADE_END_DATE = "2025-12-19"

def make_directories():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

def data_split(df, start, end):
    data = df[(df.date >= start) & (df.date < end)]
    data = data.sort_values(["date", "tic"], ignore_index=True)
    data.index = data.date.factorize()[0]
    return data

def get_baseline_stats(df):
    """
    计算基准收益 (这里使用简单的'平均持仓'作为基准)
    实际上你应该读取上证50指数(000016.SH)的真实数据做对比
    """
    print("正在计算基准收益 (平均持仓策略)...")
    unique_date = df.date.unique()
    baseline_returns = []
    
    # 假设第一天平均买入所有股票
    for i in range(len(unique_date)-1):
        # 简单计算：当天所有股票收盘价的平均涨幅
        current_date = unique_date[i]
        next_date = unique_date[i+1]
        
        current_prices = df[df.date == current_date].close.values
        next_prices = df[df.date == next_date].close.values
        
        # 简单的平均涨跌幅
        avg_pct_change = np.mean((next_prices - current_prices) / current_prices)
        baseline_returns.append(avg_pct_change)
        
    return baseline_returns

def calculate_metrics(df_account_value):
    """
    计算年化收益、夏普比率、最大回撤 
    """
    df = df_account_value.copy()
    # 计算日收益率
    df['daily_return'] = df['account_value'].pct_change(1)
    df = df.dropna()
    
    # 1. 累计收益
    final_value = df['account_value'].iloc[-1]
    initial_value = df['account_value'].iloc[0]
    cumulative_return = (final_value - initial_value) / initial_value
    
    # 2. 年化收益 (假设一年252个交易日)
    annual_return = df['daily_return'].mean() * 252
    
    # 3. 年化波动率
    annual_volatility = df['daily_return'].std() * np.sqrt(252)
    
    # 4. 夏普比率 (无风险利率假设为 3%)
    risk_free_rate = 0.03
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
    
    # 5. 最大回撤
    df['cum_return'] = (1 + df['daily_return']).cumprod()
    df['cum_max'] = df['cum_return'].cummax()
    df['drawdown'] = df['cum_max'] - df['cum_return']
    df['drawdown_pct'] = df['drawdown'] / df['cum_max']
    max_drawdown = df['drawdown_pct'].max()
    
    return {
        "Cumulative Return": f"{cumulative_return*100:.2f}%",
        "Annual Return": f"{annual_return*100:.2f}%",
        "Annual Volatility": f"{annual_volatility*100:.2f}%",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown*100:.2f}%"
    }

def run_backtest(agent_name, model_class, df_trade, stock_dim):
    """运行单个模型的历史回测"""
    print(f"--> 正在回测: {agent_name} ...")
    
    # 1. 初始化环境
    env_kwargs = {
        "df": df_trade,
        "stock_dim": stock_dim,
        "hmax": 100,
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.0003,
        "stamp_duty_pct": 0.0005,
        "reward_scaling": 1e-4,
        "turbulence_threshold": None  # 回测时可以开启风控，暂且设为None观察原始表现
    }
    env = StockTradingEnv(**env_kwargs)
    
    # 2. 加载模型
    model_path = f"{TRAINED_MODEL_DIR}/agent_{agent_name.lower()}"
    if not os.path.exists(model_path + ".zip"):
        print(f"❌ 找不到模型文件: {model_path}.zip")
        return None
        
    model = model_class.load(model_path)
    
    # 3. 循环预测
    obs, info = env.reset()
    done = False
    
    while not done:
        # deterministic=True 表示使用确定性策略（不加噪声），这对回测很重要
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = env.step(action)
        
    # 4. 获取资金曲线
    df_account = env.save_asset_memory()
    return df_account

def main():
    make_directories()
    
    print("读取数据...")
    df = pd.read_csv(DATA_PATH)
    
    # 切分测试集
    trade_df = data_split(df, TRADE_START_DATE, TRADE_END_DATE)
    if len(trade_df) == 0:
        print("❌ 测试集为空，请检查日期。")
        return
        
    print(f"测试集样本量: {len(trade_df)}")
    stock_dimension = len(trade_df.tic.unique())
    
    # --- 1. 分别回测三个模型 ---
    results = {}
    
    # PPO
    results['PPO'] = run_backtest('PPO', PPO, trade_df, stock_dimension)
    # A2C
    results['A2C'] = run_backtest('A2C', A2C, trade_df, stock_dimension)
    # DDPG
    results['DDPG'] = run_backtest('DDPG', DDPG, trade_df, stock_dimension)
    
    # --- 2. 计算基准 (Baseline) ---
    # 构造一个简单的基准资金曲线 (Index Benchmark)
    baseline_returns = get_baseline_stats(trade_df)
    
    # 重构基准的 DataFrame
    if results['PPO'] is not None:
        base_dates = results['PPO']['date'].values[:-1] # 对齐长度
        # 初始资金 1,000,000
        baseline_values = [1000000]
        for r in baseline_returns:
            baseline_values.append(baseline_values[-1] * (1 + r))
            
        # 截断到相同长度
        min_len = min(len(base_dates), len(baseline_values))
        df_baseline = pd.DataFrame({
            'date': base_dates[:min_len], 
            'account_value': baseline_values[:min_len]
        })
        results['Baseline (Avg)'] = df_baseline

    # --- 3. 打印指标 & 画图 ---
    plt.figure(figsize=(12, 6))
    
    print("\n====== 回测性能指标 (2024-2025) ======")
    for agent_name, df_res in results.items():
        if df_res is None: continue
        
        # 计算指标
        metrics = calculate_metrics(df_res)
        print(f"[{agent_name}]")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print("-" * 20)
        
        # 画图
        plt.plot(pd.to_datetime(df_res['date']), df_res['account_value'], label=agent_name)
        
        # 保存 CSV 结果
        if agent_name != 'Baseline (Avg)':
            df_res.to_csv(f"{RESULTS_DIR}/account_value_{agent_name}.csv", index=False)

    plt.title(f"A-Share Stock Trading Agents Backtest ({TRADE_START_DATE} to {TRADE_END_DATE})")
    plt.xlabel("Date")
    plt.ylabel("Account Value (RMB)")
    plt.legend()
    plt.grid(True)
    
    plot_path = f"{RESULTS_DIR}/backtest_result.png"
    plt.savefig(plot_path)
    print(f"\n回测完成！结果图已保存至: {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()