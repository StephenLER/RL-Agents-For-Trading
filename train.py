import pandas as pd
import numpy as np
import os
import time
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise

# 导入你刚才写的环境类
# 务必确保 env_stocktrading.py 和 train.py 在同一个文件夹下
from env_stocktrading import StockTradingEnv

# --- 配置参数 ---
DATA_PATH = "./data/final_rl_data.csv"
TRAINED_MODEL_DIR = "./trained_models"
TENSORBOARD_LOG_DIR = "./tensorboard_log"

# 切分日期 (根据你的数据范围 2018-01-01 到 2025-12-19)
TRAIN_START_DATE = "2018-01-01"
TRAIN_END_DATE = "2023-12-31"   # 训练集: 2018-2023
TRADE_START_DATE = "2024-01-01" 
TRADE_END_DATE = "2025-12-19"   # 测试集: 2024-2025

def make_directories():
    if not os.path.exists(TRAINED_MODEL_DIR):
        os.makedirs(TRAINED_MODEL_DIR)
    if not os.path.exists(TENSORBOARD_LOG_DIR):
        os.makedirs(TENSORBOARD_LOG_DIR)

def data_split(df, start, end):
    """
    数据切分辅助函数
    """
    data = df[(df.date >= start) & (df.date < end)]
    data = data.sort_values(["date", "tic"], ignore_index=True)
    data.index = data.date.factorize()[0]
    return data

def train_ppo(env_train, total_timesteps=100000):
    """训练 PPO 代理"""
    print("====== 正在训练 PPO Agent ======")
    start_time = time.time()
    
    # PPO 参数通常比较稳健，这里使用常用默认值
    model = PPO("MlpPolicy", 
                env_train, 
                ent_coef=0.01, # 增加熵系数鼓励探索
                learning_rate=0.00025, 
                batch_size=128,
                verbose=1, 
                tensorboard_log=TENSORBOARD_LOG_DIR)
    
    model.learn(total_timesteps=total_timesteps)
    model.save(f"{TRAINED_MODEL_DIR}/agent_ppo")
    
    print(f"PPO 训练完成，耗时: {(time.time() - start_time)/60:.2f} 分钟")
    return model

def train_a2c(env_train, total_timesteps=100000):
    """训练 A2C 代理"""
    print("====== 正在训练 A2C Agent ======")
    start_time = time.time()
    
    model = A2C("MlpPolicy", 
                env_train, 
                ent_coef=0.01,
                verbose=1, 
                tensorboard_log=TENSORBOARD_LOG_DIR)
    
    model.learn(total_timesteps=total_timesteps)
    model.save(f"{TRAINED_MODEL_DIR}/agent_a2c")
    
    print(f"A2C 训练完成，耗时: {(time.time() - start_time)/60:.2f} 分钟")
    return model

def train_ddpg(env_train, total_timesteps=50000):
    """训练 DDPG 代理"""
    print("====== 正在训练 DDPG Agent ======")
    start_time = time.time()
    
    # DDPG 需要为动作空间添加噪声以进行探索
    n_actions = env_train.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    model = DDPG("MlpPolicy", 
                 env_train, 
                 action_noise=action_noise,
                 learning_rate=0.0005,
                 batch_size=128,
                 buffer_size=50000, # 经验回放池大小
                 verbose=1, 
                 tensorboard_log=TENSORBOARD_LOG_DIR)
    
    model.learn(total_timesteps=total_timesteps)
    model.save(f"{TRAINED_MODEL_DIR}/agent_ddpg")
    
    print(f"DDPG 训练完成，耗时: {(time.time() - start_time)/60:.2f} 分钟")
    return model

def main():
    make_directories()
    
    # 1. 读取并预处理数据
    print(f"正在读取数据: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误: 找不到数据文件 {DATA_PATH}，请先运行 02_process_data.py")
        return

    df = pd.read_csv(DATA_PATH)
    
    # 计算股票数量 (Stock Dimension)
    stock_dimension = len(df.tic.unique())
    state_space = 1 + 2 * stock_dimension + 4 * stock_dimension 
    print(f"股票数量: {stock_dimension}")
    print(f"状态空间维度: {state_space}")
    
    # 2. 切分训练集
    train_df = data_split(df, TRAIN_START_DATE, TRAIN_END_DATE)
    print(f"训练集样本量: {len(train_df)}")
    
    if len(train_df) == 0:
        print("❌ 错误: 训练集为空，请检查日期范围。")
        return

    # 3. 初始化训练环境
    env_train_kwargs = {
        "df": train_df,
        "stock_dim": stock_dimension,
        "hmax": 100, 
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.0003, # 佣金万三
        "stamp_duty_pct": 0.0005,       # 印花税万五
        "reward_scaling": 1e-4, 
        "turbulence_threshold": None 
    }
    
    print("正在构建向量化环境...")
    # 使用 DummyVecEnv 封装 lambda 函数，这是 Stable-Baselines3 的标准用法
    env_train = DummyVecEnv([lambda: StockTradingEnv(**env_train_kwargs)])

    # 4. 开始训练三个模型
    # 建议步数: 测试用 10000, 实际训练建议 50000+
    TRAIN_STEPS = 50000 
    
    print("-" * 30)
    ppo_model = train_ppo(env_train, total_timesteps=TRAIN_STEPS)
    print("-" * 30)
    a2c_model = train_a2c(env_train, total_timesteps=TRAIN_STEPS)
    print("-" * 30)
    ddpg_model = train_ddpg(env_train, total_timesteps=TRAIN_STEPS)
    
    print("=" * 30)
    print("所有模型训练完毕！模型已保存至 ./trained_models 文件夹")
    print("请运行下一步的回测脚本进行验证。")

if __name__ == "__main__":
    main()