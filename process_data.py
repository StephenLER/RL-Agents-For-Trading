import pandas as pd
import numpy as np
import os

# 检查 talib 是否安装
try:
    import talib
except ImportError:
    print("❌ 错误: 未找到 ta-lib 库。")
    exit()

def clean_incomplete_stocks(df, threshold=0.95):
    """
    清洗数据：剔除数据严重缺失的股票
    """
    print("正在进行数据完整性检查...")
    tic_counts = df['tic'].value_counts()
    max_len = tic_counts.max()
    print(f"标准数据长度 (Max): {max_len} 行")
    
    valid_tics = tic_counts[tic_counts >= max_len * threshold].index.tolist()
    dropped_tics = tic_counts[tic_counts < max_len * threshold].index.tolist()
    
    print(f"保留股票数量: {len(valid_tics)}")
    if dropped_tics:
        print(f"❌ 剔除 {len(dropped_tics)} 只数据缺失严重的股票")
    
    df_clean = df[df['tic'].isin(valid_tics)].copy()
    return df_clean

def align_data(df):
    """
    【核心修复】数据对齐
    确保每个日期都有所有的股票，缺失的用前值填充 (ffill)
    """
    print("正在对齐数据 (处理停牌和缺失值)...")
    
    # 获取所有唯一的日期和股票代码
    unique_dates = df.date.unique()
    unique_tics = df.tic.unique()
    
    # 创建完整的 MultiIndex (日期 x 股票)
    full_idx = pd.MultiIndex.from_product(
        [unique_dates, unique_tics], names=["date", "tic"]
    )
    
    # 设置索引并重新索引 (Reindex) 会自动生成缺失的行，值为 NaN
    df = df.set_index(["date", "tic"])
    df = df.reindex(full_idx)
    
    # 排序
    df = df.sort_index()
    
    # 填充缺失值逻辑：
    # 1. 价格/指标：用该股票前一天的数据填充 (模拟停牌，价格不变)
    # 2. 交易量：停牌期间设为 0
    print("正在填充缺失数据...")
    df = df.groupby(level="tic").ffill() # 前向填充 (核心)
    df = df.groupby(level="tic").bfill() # 后向填充 (处理开头缺失)
    df = df.fillna(0) # 兜底
    
    df = df.reset_index()
    return df

def add_technical_indicators(df):
    """计算 MACD, RSI, CCI, ADX"""
    print("正在计算技术指标...")
    processed_dfs = []
    
    for ticker, group in df.groupby('tic'):
        group = group.sort_values('date').copy()
        # 必须重置索引，否则 talib 计算可能会错位
        group = group.reset_index(drop=True)
        
        close = group['close'].values.astype(float)
        high = group['high'].values.astype(float)
        low = group['low'].values.astype(float)
        
        try:
            # 填充极少量的 NaN 防止 talib 报错
            if np.isnan(close).any():
                group = group.ffill().bfill().fillna(0)
                close = group['close'].values
                high = group['high'].values
                low = group['low'].values

            group['macd'], _, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            group['rsi'] = talib.RSI(close, timeperiod=14)
            group['cci'] = talib.CCI(high, low, close, timeperiod=14)
            group['adx'] = talib.ADX(high, low, close, timeperiod=14)
            
            processed_dfs.append(group)
            
        except Exception as e:
            print(f"⚠️ {ticker} 指标计算错误: {e}")
            continue
            
    return pd.concat(processed_dfs)

def calculate_turbulence(df, window_size=252):
    """计算湍流指数"""
    print(f"正在计算湍流指数 (窗口: {window_size})...")
    df = df.copy()
    df['return'] = df.groupby('tic')['close'].pct_change()
    
    pivot_returns = df.pivot(index='date', columns='tic', values='return')
    
    turbulence_index = []
    dates = pivot_returns.index
    initial_zeros = [0] * window_size
    turbulence_index.extend(initial_zeros)
    
    for i in range(window_size, len(dates)):
        current_history = pivot_returns.iloc[i-window_size : i]
        current_return = pivot_returns.iloc[i]
        
        mu = current_history.mean()
        sigma = current_history.cov()
        diff = current_return - mu
        
        try:
            # 只计算非NaN的有效列
            valid_cols = current_history.columns[current_history.notna().any()]
            if len(valid_cols) < 2:
                turbulence_index.append(0)
                continue
                
            sigma_filtered = sigma.loc[valid_cols, valid_cols].fillna(0)
            diff_filtered = diff[valid_cols].fillna(0)
            
            sigma_inv = np.linalg.pinv(sigma_filtered)
            turbulence = diff_filtered.dot(sigma_inv).dot(diff_filtered.T)
            turbulence_index.append(turbulence)
        except:
            turbulence_index.append(0)
            
    turbulence_df = pd.DataFrame({'date': dates, 'turbulence': turbulence_index})
    df = df.merge(turbulence_df, on='date', how='left')
    return df

def main():
    input_path = 'data/raw_stock_data.csv'
    output_path = 'data/final_rl_data.csv'
    
    if not os.path.exists(input_path):
        print(f"❌ 未找到 {input_path}")
        return
    
    print(f"读取原始数据: {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. 基础过滤：去除成交量为0的（停牌），但后面我们会补回来
    df = df[df['volume'] > 0]
    
    # 2. 剔除严重缺失的股票
    df = clean_incomplete_stocks(df, threshold=0.90)
    
    # 3. 【关键步骤】数据对齐补全
    # 这一步保证了每天的股票数量完全一致，防止 Shape Mismatch 错误
    df = align_data(df)

    # 4. 计算指标
    df = add_technical_indicators(df)
    
    # 5. 计算湍流
    df = calculate_turbulence(df, window_size=252)
    
    # 6. 最终清洗
    # 去除指标计算产生的头部 NaN
    df.dropna(inplace=True)
    df = df[df['turbulence'] != 0]
    
    # 再次按日期排序，确保环境读取顺序正确
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    df.to_csv(output_path, index=False)
    print("-" * 30)
    print("✅ 数据预处理完成！")
    print(f"最终样本量: {df.shape[0]}")
    print(f"股票数量: {df['tic'].nunique()} (请确保所有日期此数量一致)")
    
    # 验证完整性
    dates_count = df.groupby('date')['tic'].count()
    if dates_count.min() == dates_count.max():
        print(f"✅ 数据验证通过：每一天都有 {dates_count.max()} 只股票")
    else:
        print(f"⚠️ 数据验证警告：部分日期股票数量不一致 (Min: {dates_count.min()}, Max: {dates_count.max()})")

if __name__ == "__main__":
    main()