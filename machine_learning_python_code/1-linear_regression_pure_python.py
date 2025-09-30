import numpy as np
import pandas as pd
import os
import pickle
import random

# 设置随机种子，保证结果可复现
random.seed(42)
np.random.seed(42)

# 1. 生成模拟数据集 - 房屋面积与价格的关系
def generate_dataset(n_samples=1000):
    """生成模拟的房屋面积与价格数据集"""
    # 房屋面积 (平方米)
    area = np.random.uniform(50, 200, n_samples)
    
    # 房间数量
    rooms = np.random.randint(1, 5, n_samples)
    
    # 房龄 (年)
    age = np.random.uniform(0, 30, n_samples)
    
    # 基础价格 = 面积 * 8000 + 房间数 * 50000 - 房龄 * 2000 + 常数项
    base_price = area * 8000 + rooms * 50000 - age * 2000 + 300000
    
    # 添加噪声
    noise = np.random.normal(0, 100000, n_samples)
    price = base_price + noise
    
    # 创建DataFrame
    df = pd.DataFrame({
        'area': area,
        'rooms': rooms,
        'age': age,
        'price': price
    })
    
    # 随机添加一些缺失值
    for col in df.columns:
        df.loc[np.random.choice(df.index, size=int(n_samples*0.05)), col] = np.nan
    
    # 随机添加一些异常值
    outliers = np.random.choice(df.index, size=int(n_samples*0.03))
    df.loc[outliers, 'price'] *= np.random.uniform(1.5, 3, len(outliers))
    
    return df

# 2. 数据清洗函数
def clean_data(df):
    """数据清洗流程"""
    # 复制数据以避免修改原始数据
    cleaned_df = df.copy()
    
    # 处理缺失值
    # 对于数值型特征，使用中位数填充
    for col in ['area', 'age', 'price']:
        median = cleaned_df[col].median()
        cleaned_df[col].fillna(median, inplace=True)
    
    # 对于离散型特征(房间数)，使用众数填充
    mode = cleaned_df['rooms'].mode()[0]
    cleaned_df['rooms'].fillna(mode, inplace=True)
    
    # 处理异常值 - 使用IQR方法
    def remove_outliers(df, column):
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # 对价格进行异常值处理
    cleaned_df = remove_outliers(cleaned_df, 'price')
    
    # 确保房间数为整数
    cleaned_df['rooms'] = cleaned_df['rooms'].astype(int)
    
    return cleaned_df

# 3. 自定义数据集划分函数
def train_val_test_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    将数据集划分为训练集、验证集和测试集
    test_size: 测试集占比
    val_size: 验证集占比
    """
    # 设置随机种子
    if random_state is not None:
        np.random.seed(random_state)
    
    # 获取样本数量
    n_samples = len(X)
    
    # 生成索引并打乱
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # 计算划分边界
    test_split = int(n_samples * (1 - test_size - val_size))
    val_split = int(n_samples * (1 - val_size))
    
    # 划分数据集
    X_train = X[indices[:test_split]]
    y_train = y[indices[:test_split]]
    X_val = X[indices[test_split:val_split]]
    y_val = y[indices[test_split:val_split]]
    X_test = X[indices[val_split:]]
    y_test = y[indices[val_split:]]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# 4. 特征标准化
class StandardScaler:
    """自定义标准化器"""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X):
        """计算均值和标准差"""
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # 防止除以零
        self.scale_[self.scale_ == 0] = 1.0
        return self
        
    def transform(self, X):
        """标准化数据"""
        return (X - self.mean_) / self.scale_
        
    def save(self, path):
        """保存标准化器参数"""
        with open(path, 'wb') as f:
            pickle.dump({'mean_': self.mean_, 'scale_': self.scale_}, f)
    
    @classmethod
    def load(cls, path):
        """加载标准化器"""
        scaler = cls()
        with open(path, 'rb') as f:
            params = pickle.load(f)
        scaler.mean_ = params['mean_']
        scaler.scale_ = params['scale_']
        return scaler

# 5. 评估指标函数
def mean_squared_error(y_true, y_pred):
    """计算均方误差"""
    return np.mean((y_true - y_pred) **2)

def root_mean_squared_error(y_true, y_pred):
    """计算均方根误差"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r2_score(y_true, y_pred):
    """计算决定系数R²"""
    ss_total = np.sum((y_true - np.mean(y_true))** 2)
    ss_residual = np.sum((y_true - y_pred) **2)
    return 1 - (ss_residual / ss_total)

# 6. 线性回归模型
class LinearRegression:
    """线性回归模型，使用梯度下降法优化"""
    def __init__(self, learning_rate=0.01, epochs=1000, batch_size=32, 
                 early_stopping_patience=20, verbose=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = {'train': [], 'val': []}
        
    def _compute_loss(self, X, y):
        """计算均方误差损失"""
        n_samples = len(y)
        y_pred = self.predict(X)
        loss = (1 / n_samples) * np.sum((y_pred - y) **2)
        return loss
        
    def _compute_gradients(self, X, y):
        """计算梯度"""
        n_samples = len(y)
        y_pred = self.predict(X)
        
        # 计算梯度
        dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (2 / n_samples) * np.sum(y_pred - y)
        
        return dw, db
        
    def _gradient_descent_step(self, X, y):
        """梯度下降一步更新"""
        dw, db = self._compute_gradients(X, y)
        
        # 更新参数
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        n_samples, n_features = X_train.shape
        
        # 初始化参数
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        
        # 早停机制相关变量
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 训练迭代
        for epoch in range(self.epochs):
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # 批量梯度下降
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                self._gradient_descent_step(X_batch, y_batch)
            
            # 计算训练损失
            train_loss = self._compute_loss(X_train, y_train)
            self.loss_history['train'].append(train_loss)
            
            # 计算验证损失
            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self._compute_loss(X_val, y_val)
                self.loss_history['val'].append(val_loss)
            
            # 打印训练信息
            if self.verbose and (epoch + 1) % 100 == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{self.epochs}, "
                          f"Train Loss: {train_loss:.2f}, "
                          f"Val Loss: {val_loss:.2f}")
                else:
                    print(f"Epoch {epoch+1}/{self.epochs}, "
                          f"Train Loss: {train_loss:.2f}")
            
            # 早停机制
            if X_val is not None and y_val is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = self.weights.copy()
                    best_bias = self.bias
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        if self.verbose:
                            print(f"早停在第 {epoch+1} 轮")
                        self.weights = best_weights
                        self.bias = best_bias
                        break
        
        return self
    
    def predict(self, X):
        """预测"""
        return np.dot(X, self.weights) + self.bias
    
    def save_model(self, path):
        """保存模型参数"""
        model_params = {
            'weights': self.weights,
            'bias': self.bias
        }
        with open(path, 'wb') as f:
            pickle.dump(model_params, f)
    
    @classmethod
    def load_model(cls, path):
        """加载模型参数"""
        with open(path, 'rb') as f:
            model_params = pickle.load(f)
        
        model = cls()
        model.weights = model_params['weights']
        model.bias = model_params['bias']
        return model

# 7. 模型训练与评估主函数
def train_and_evaluate():
    # 创建模型保存目录
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 生成数据集
    print("生成数据集...")
    df = generate_dataset(1000)
    print(f"原始数据集形状: {df.shape}")
    
    # 数据清洗
    print("数据清洗中...")
    cleaned_df = clean_data(df)
    print(f"清洗后数据集形状: {cleaned_df.shape}")
    
    # 划分特征和目标变量
    X = cleaned_df[['area', 'rooms', 'age']].values
    y = cleaned_df['price'].values
    
    # 数据集划分：70%训练集，15%验证集，15%测试集
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, test_size=0.15, val_size=0.15, random_state=42
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 保存标准化器
    scaler.save('models/scaler.pkl')
    
    # 初始化并训练模型
    print("开始训练模型...")
    model = LinearRegression(
        learning_rate=0.01,
        epochs=2000,
        batch_size=32,
        early_stopping_patience=30,
        verbose=True
    )
    
    model.fit(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # 在测试集上评估模型
    print("\n在测试集上评估模型...")
    y_pred = model.predict(X_test_scaled)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"测试集均方误差 (MSE): {mse:.2f}")
    print(f"测试集均方根误差 (RMSE): {rmse:.2f}")
    print(f"决定系数 (R²): {r2:.4f}")
    
    # 保存模型
    model.save_model('models/linear_regression_model.pkl')
    print("模型已保存到 'models' 目录")
    
    return model, scaler, mse, rmse, r2

if __name__ == "__main__":
    # 训练并评估模型
    model, scaler, mse, rmse, r2 = train_and_evaluate()
    
    # 模型应用示例
    print("\n模型应用示例:")
    # 假设一个新的房屋数据
    new_house = np.array([[120, 3, 5]])  # 面积120平方米, 3个房间, 房龄5年
    
    # 特征标准化
    new_house_scaled = scaler.transform(new_house)
    
    # 预测价格
    predicted_price = model.predict(new_house_scaled)[0]
    print(f"对于面积{new_house[0][0]}平方米, {new_house[0][1]}个房间, 房龄{new_house[0][2]}年的房屋，")
    print(f"预测价格为: {predicted_price:.2f}元")
    