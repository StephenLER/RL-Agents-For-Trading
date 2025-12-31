import numpy as np
import pandas as pd
import gymnasium as gym  # 使用新版 gym (gymnasium)
from gymnasium import spaces
import matplotlib.pyplot as plt

class StockTradingEnv(gym.Env):
    """
    A股股票交易环境，基于论文复现
    State Space: [余额] + [每只股票收盘价] + [每只股票持仓] + [每只股票技术指标...]
    Action Space: -1 到 1 的连续数值 (表示卖出/买入的比例)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, stock_dim, initial_amount=1000000, 
                 hmax=100, turbulence_threshold=None, 
                 transaction_cost_pct=0.0003, stamp_duty_pct=0.0005, 
                 reward_scaling=1e-4, tech_indicator_list=['macd', 'rsi', 'cci', 'adx']):
        
        self.df = df
        self.stock_dim = stock_dim # 股票数量
        self.initial_amount = initial_amount # 初始资金
        self.hmax = hmax # 单次最大交易手数 (注意：这里我们定义为'手'，即100股)
        self.turbulence_threshold = turbulence_threshold # 湍流阈值
        
        # 费率：佣金万三，印花税万五(卖出收)
        self.transaction_cost_pct = transaction_cost_pct 
        self.stamp_duty_pct = stamp_duty_pct 
        
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list

        # 动作空间: [-1, 1] 的连续向量，维度 = 股票数量
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,))

        # 状态空间: 
        # 1 (余额) + stock_dim (股价) + stock_dim (持仓) + stock_dim * 4 (技术指标)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(1 + self.stock_dim * (2 + len(self.tech_indicator_list)),)
        )

        # 初始化数据指针
        self.data = self.df.loc[self.df.date == self.df.date.unique()[0]]
        self.terminal = False # 是否结束
        self.day = 0 
        
        # 保存初始化状态
        self.state = self._initiate_state()

        # 初始化资产记录
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.date_memory = [self._get_date()]

    def _sell_stock(self, index, action):
        """
        执行卖出逻辑 (A股规则: 卖出收印花税 + 佣金)
        """
        # action 是 [-1, 0]，转换为卖出手数
        # 假设 action=-1 代表卖出 hmax 手
        if self.state[index + self.stock_dim + 1] > 0: # 检查持仓是否 > 0
            # 卖出的数量 = abs(action) * hmax * 100 (股)
            # 取整为100的倍数
            num_shares = int(abs(action) * self.hmax) * 100
            
            # 检查持仓够不够卖
            shares_held = self.state[index + self.stock_dim + 1]
            num_shares = min(num_shares, shares_held)

            if num_shares > 0:
                price = self.state[index + 1]
                amount = price * num_shares
                
                # 成本 = 佣金 + 印花税
                cost = amount * self.transaction_cost_pct + amount * self.stamp_duty_pct
                
                self.state[0] += (amount - cost) # 余额增加
                self.state[index + self.stock_dim + 1] -= num_shares # 持仓减少
                self.cost += cost
                self.trades += 1
        else:
            pass

    def _buy_stock(self, index, action):
        """
        执行买入逻辑 (A股规则: 买入仅收佣金)
        """
        # action 是 [0, 1]
        # 买入数量 = action * hmax * 100
        num_shares = int(action * self.hmax) * 100
        price = self.state[index + 1]
        
        # 估算成本，防止资金不足
        amount = price * num_shares
        cost = amount * self.transaction_cost_pct
        
        if self.state[0] > (amount + cost): # 如果余额足够
            self.state[0] -= (amount + cost)
            self.state[index + self.stock_dim + 1] += num_shares
            self.cost += cost
            self.trades += 1
        else:
            # 资金不足，能买多少买多少
            available_amount = self.state[0]
            # 预留一点钱付手续费
            max_can_buy = int(available_amount / (price * (1 + self.transaction_cost_pct)))
            # 向下取整到100股
            num_shares = (max_can_buy // 100) * 100
            
            if num_shares > 0:
                amount = price * num_shares
                cost = amount * self.transaction_cost_pct
                self.state[0] -= (amount + cost)
                self.state[index + self.stock_dim + 1] += num_shares
                self.cost += cost
                self.trades += 1

    def step(self, actions):
        """
        Agent 与环境交互的核心步骤
        """
        self.terminal = self.day >= len(self.df.date.unique()) - 1
        
        if self.terminal:
            # 交易结束
            # print(f"Episode End. Total Asset: {self.state[0] + sum(np.array(self.state[1:(self.stock_dim+1)]) * np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))}")
            return self.state, 0, self.terminal, False, {}

        else:
            # 1. 获取动作
            # 动作乘以 hmax (放大比例)
            actions = actions * self.hmax 
            
            # 获取当前的总资产 (作为计算 reward 的基准)
            # 资产 = 余额 + sum(股价 * 持仓)
            begin_total_asset = self.state[0] + \
                sum(np.array(self.state[1:(self.stock_dim+1)]) * \
                    np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))

            # 2. 检查湍流指数 (Turbulence Risk Control) 
            # 如果湍流指数超过阈值，强制平仓 (Sell all)
            turbulence_val = self.data['turbulence'].values[0]
            if self.turbulence_threshold is not None:
                if turbulence_val > self.turbulence_threshold:
                    actions = np.array([-1] * self.stock_dim) # 全力卖出
            
            # 3. 执行交易
            self.cost = 0
            self.trades = 0
            
            # argsort 排序，先卖后买，腾出资金
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self._sell_stock(index, actions[index])

            for index in buy_index:
                self._buy_stock(index, actions[index])

            # 4. 更新时间步 t -> t+1
            self.day += 1
            self.data = self.df.loc[self.df.date == self.df.date.unique()[self.day]]
            
            # 更新 State (新的价格和技术指标)
            self.state = self._update_state()

            # 5. 计算 Reward
            end_total_asset = self.state[0] + \
                sum(np.array(self.state[1:(self.stock_dim+1)]) * \
                    np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            
            # Reward = 资产增值 
            reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(reward)
            
            # 缩放 Reward 有利于神经网络收敛
            reward = reward * self.reward_scaling

        return self.state, reward, self.terminal, False, {}

    def reset(self, seed=None):
        """重置环境"""
        super().reset(seed=seed)
        self.day = 0
        self.data = self.df.loc[self.df.date == self.df.date.unique()[0]]
        self.state = self._initiate_state()
        self.asset_memory = [self.initial_amount]
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []
        self.date_memory = [self._get_date()]
        return self.state, {}

    def _initiate_state(self):
        """初始化状态向量"""
        if self.initial_amount is None:
             self.initial_amount = 1000000
             
        # 1. 余额
        state = [self.initial_amount] 
        
        # 2. 初始股价
        state += self.data.close.values.tolist()
        
        # 3. 初始持仓 (全为0)
        state += [0] * self.stock_dim
        
        # 4. 技术指标
        for tech in self.tech_indicator_list:
            state += self.data[tech].values.tolist()
            
        return state

    def _update_state(self):
        """更新状态向量"""
        # 1. 余额 (保留上一时刻的余额，因为 step 中已经更新了买卖后的余额)
        state = [self.state[0]]
        
        # 2. 新的股价
        state += self.data.close.values.tolist()
        
        # 3. 持仓 (step 中已更新)
        state += list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)])
        
        # 4. 新的技术指标
        for tech in self.tech_indicator_list:
            state += self.data[tech].values.tolist()
            
        return state

    def _get_date(self):
        return self.data.date.unique()[0]

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        return pd.DataFrame({'date':date_list, 'account_value':asset_list})