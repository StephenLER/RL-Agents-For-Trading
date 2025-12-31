# 01_fetch_data.py
import pandas as pd
from WindPy import w
import os
import datetime

def fetch_wind_data(start_date, end_date, index_code="000016.SH"):
    """
    连接 Wind 并下载原始数据
    """
    # 1. 连接 Wind
    print("正在连接 Wind API...")
    w.start()
    if not w.isconnected():
        print("❌ Wind 连接失败，请检查终端是否登录")
        return None

    print(f"✅ Wind 连接成功. 目标区间: {start_date} 至 {end_date}")

    # 2. 获取成分股 (逻辑优化)
    # 自动修正日期：如果 end_date 是非交易日或未来日期，尝试向前修正
    # 但最稳妥的是直接使用最近的一个交易日
    offset_res = w.tdaysoffset(0, end_date, "")
    
    if offset_res.ErrorCode == 0 and offset_res.Data:
        search_date = offset_res.Data[0][0].strftime('%Y-%m-%d')
        # 再次检查：如果修正后的日期还是超过了今天(未来)，则强制使用今天
        today = datetime.date.today().strftime('%Y-%m-%d')
        if search_date > today:
            print(f"⚠️ 警告: 请求日期 {search_date} 在未来，强制调整为今天 {today}")
            search_date = today
    else:
        search_date = end_date
        
    print(f"正在获取 {index_code} 在 {search_date} 的成分股...")
    
    # --- 关键修改：使用 indexconstituent 而不是 sectorconstituent ---
    # --- 关键修改：参数名 sectorid 改为 windcode ---
    sector_data = w.wset("indexconstituent", f"date={search_date};windcode={index_code}")
    
    if sector_data.ErrorCode != 0:
        print(f"❌ 获取成分股报错，错误码: {sector_data.ErrorCode}")
        return None
        
    if not sector_data.Data:
        print(f"❌ 获取成分股成功但数据为空。请检查：\n1. 您的Wind账号是否有 {index_code} 权限\n2. 日期 {search_date} 是否有效")
        return None
        
    ticker_list = sector_data.Data[1] # Data[1] 是代码列表
    print(f"✅ 获取到 {len(ticker_list)} 只成分股，开始下载日频行情...")

    # 3. 循环下载数据
    all_data = []
    
    for i, ticker in enumerate(ticker_list):
        print(f"[{i+1}/{len(ticker_list)}] 下载 {ticker} ...", end="\r")
        
        # 下载行情
        error_code, data = w.wsd(
            ticker,
            "open,high,low,close,volume",
            start_date,
            end_date,
            "PriceAdj=B", # 后复权
            usedf=True
        )
        
        if error_code == 0 and not data.empty:
            df = data
            df.index.name = 'date'
            df.reset_index(inplace=True)
            df['tic'] = ticker
            all_data.append(df)
        else:
            if error_code != 0:
                print(f"\n⚠️ {ticker} 下载异常: Error {error_code}")

    print("\n下载完成，正在合并数据...")
    
    if not all_data:
        print("❌ 未获取到任何数据")
        return None

    # 4. 合并与保存
    final_df = pd.concat(all_data, axis=0)
    final_df.columns = [c.lower() for c in final_df.columns]
    
    # 简单的清洗
    final_df.drop_duplicates(subset=['date', 'tic'], inplace=True)
    
    if not os.path.exists('data'):
        os.makedirs('data')
        
    output_path = 'data/raw_stock_data.csv'
    final_df.to_csv(output_path, index=False)
    print(f"✅ 原始数据已保存至: {output_path}")
    print(f"数据形状: {final_df.shape}")
    
    return final_df

if __name__ == "__main__":
    # --- 请根据您当前的真实时间调整这里 ---
    START = "2018-01-01"
    
    # 建议设置为一个确定的过去日期，或者使用 today
    # 比如设置为 2024 年底，或者 datetime.date.today()
    # 只有当您的 Wind 终端真正处于 2025 年时，才能写 2025
    END = "2025-12-19"  
    
    print(f"设定回测区间: {START} 至 {END}")
    fetch_wind_data(START, END)