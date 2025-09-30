'''
这个线性回归模型实现了工程化应用所需的完整流程，主要特点包括：
>数据生成：模拟了房屋价格数据集，包含面积、房间数和房龄三个特征，更贴近实际应用场景，并加入了噪声和异常值。
>数据清洗：
    >处理缺失值：使用中位数填充数值型特征，众数填充离散型特征
    >处理异常值：使用 IQR 方法检测并移除价格异常值
    >数据类型转换：确保房间数为整数类型
>模型设计：
    >使用了带有两个隐藏层的神经网络作为回归模型，增加了模型的拟合能力
    >选择 Adam 优化器，这是在实际工程中表现优异的优化方法
    >添加了学习率调度器和早停策略，防止过拟合并提高训练效率
>数据集划分：采用 7:1.5:1.5 的比例划分训练集、验证集和测试集，符合机器学习最佳实践。
>评估与可视化：
    >使用 MSE、RMSE 和 R² 作为评估指标
    >生成多种可视化图表，包括相关性矩阵、特征散点图、训练损失曲线、预测对比图和残差图
    >结果保存到 plots 目录，便于后续分析
>工程化考虑：
    >模型训练结果可复现（设置随机种子）
    >保存训练好的模型，便于部署和复用
    >提供了模型应用示例，展示如何使用训练好的模型进行预测
运行此代码后，会生成一个可用的线性回归模型，可用于预测房屋价格，所有可视化结果会保存在 plots 文件夹中，模型文件保存为 h5 格式。
'''
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import os

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 生成模拟数据集 - 假设是房屋面积、房间数、房龄与价格的关系
def generate_dataset(n_samples=1000):
    # 房屋面积 (平方米) - 生成1000个样本，面积在50到200之间，均匀分布
    area = np.random.uniform(50, 200, n_samples)
    
    # 房间数量 - 生成1到5个房间的整数，均匀分布
    rooms = np.random.randint(1, 5, n_samples)
    
    # 房龄 (年)
    age = np.random.uniform(0, 30, n_samples)
    
    # 基础价格 = 面积 * 8000 + 房间数 * 50000 - 房龄 * 2000 + 常数项
    base_price = area * 8000 + rooms * 50000 - age * 2000 + 300000
    
    # 添加噪声（均值为0，标准差为100000的正态分布噪声）
    noise = np.random.normal(0, 100000, n_samples)
    price = base_price + noise
    
    # 创建DataFrame, 表格共有四列：面积、房间数、房龄、价格
    df = pd.DataFrame({
        'area': area,
        'rooms': rooms,
        'age': age,
        'price': price
    })
    
    # 随机添加一些缺失值 (模拟实际数据中的情况)
    for col in df.columns: # 对每一列随机添加5%的缺失值
        df.loc[np.random.choice(df.index, size=int(n_samples*0.05)), col] = np.nan
    '''
    df.index：DataFrame的行索引，表示数据集中的每一行。它是一个包含从0到n_samples-1的整数数组。
    size=int(n_samples*0.05)：计算要抽取的样本数量，这里是总样本数的5%，即int(1000*0.05)=50。
    np.random.choice(a, size, replace=False) 表示从数组a中随机抽取size个样本（不放回抽样），返回一个包含抽样结果的数组。
    其中a是要抽样的数组，size是抽取的数量，replace表示是否放回抽样（False表示不放回）。
    df.loc[row,col] = np.nan：使用loc定位到(row,col)并设为NaN，表示缺失值。
    '''
    # 随机添加一些异常值
    outliers = np.random.choice(df.index, size=int(n_samples*0.03)) # 随机选择3%的样本作为异常值
    df.loc[outliers, 'price'] *= np.random.uniform(1.5, 3, len(outliers)) # 异常值的价格 *= 1.5到3之间的随机数
    
    return df

# 2. 数据清洗函数
def clean_data(df):
    """数据清洗流程"""
    # 复制数据以避免修改原始数据
    cleaned_df = df.copy() # df.copy() 深拷贝，创建一个与df完全独立的新DataFrame，避免修改原始数据
    
    # 处理缺失值
    # 对于数值型特征，使用中位数填充
    '''
    Pandas的fillna()函数用于填充DataFrame或Series中的缺失值（NaN）。
    fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None, **kwargs)
    value：用于填充缺失值的标量值或字典。如果为字典，则字典的键是列名，值是要填充的值。
    method：填充方法，可以是'ffill'（前向填充）或'bfill'（后向填充）。
    inplace：是否在原地修改DataFrame。如果为True，则直接修改原始DataFrame；如果为False，则返回一个新的DataFrame。
    axis：指定填充的方向。0表示按列填充，1表示按行填充。默认为0。
    limit：指定填充的最大次数。如果为None，则没有限制。
    downcast：是否尝试将数据类型转换为更小的类型。默认为None。
    **kwargs：其他参数。
    '''
    for col in ['area', 'age', 'price']:
        cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    # 对于离散型特征(房间数)，使用众数填充
    cleaned_df['rooms'].fillna(cleaned_df['rooms'].mode()[0], inplace=True)
    
    # 处理异常值 - 使用IQR方法
    # 处理异常值 - 使用IQR方法
    def remove_outliers(df, column):
        """ 
        使用IQR(四分位距)方法检测并移除指定列中的异常值
        
        参数:
            df (pandas.DataFrame): 包含待处理数据的DataFrame
            column (str): 需要进行异常值处理的列名
        
        返回:
            pandas.DataFrame: 移除异常值后的新DataFrame
        
        原理:
            IQR方法基于数据的四分位数来确定异常值边界，计算公式为：
            1. 第一四分位数(Q1) = 数据的25%分位数
            2. 第三四分位数(Q3) = 数据的75%分位数
            3. 四分位距(IQR) = Q3 - Q1
            4. 异常值下限 = Q1 - 1.5 * IQR
            5. 异常值上限 = Q3 + 1.5 * IQR
            6. 处于[下限, 上限]范围外的值被判定为异常值并移除
        """
        # 计算第一四分位数（25%分位数）
        q1 = df[column].quantile(0.25)
        # 计算第三四分位数（75%分位数）
        q3 = df[column].quantile(0.75)
        # 计算四分位距
        iqr = q3 - q1
        # 计算异常值的下限
        lower_bound = q1 - 1.5 * iqr
        # 计算异常值的上限
        upper_bound = q3 + 1.5 * iqr
        # 返回过滤后的数据，保留值在正常范围内的行
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # 对价格进行异常值处理
    cleaned_df = remove_outliers(cleaned_df, 'price')
    
    # 确保房间数为整数
    cleaned_df['rooms'] = cleaned_df['rooms'].astype(int)
    
    return cleaned_df

# 3. 数据可视化与分析
def analyze_data(df):
    """数据可视化分析"""
    # 创建结果保存目录
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # 绘制相关性矩阵
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('特征相关性矩阵')
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    
    # 绘制各特征与价格的散点图
    features = ['area', 'rooms', 'age']
    for feature in features:
        plt.figure(figsize=(8, 6))
        plt.scatter(df[feature], df['price'], alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel('价格')
        plt.title(f'{feature}与价格的关系')
        plt.savefig(f'plots/{feature}_vs_price.png')
        plt.close()
    
    # 绘制价格分布
    plt.figure(figsize=(8, 6))
    sns.histplot(df['price'], kde=True)
    plt.title('价格分布')
    plt.savefig('plots/price_distribution.png')
    plt.close()

# 4. 构建线性回归模型
def build_model(input_dim):
    """构建线性回归模型"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # 输出层，预测价格
    ])
    
    # 选择Adam优化器，学习率0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss='mse',  # 均方误差损失
        metrics=['mae']  # 平均绝对误差作为评估指标
    )
    
    return model

# 5. 模型训练与评估
def train_and_evaluate():
    # 生成数据集
    print("生成数据集...")
    df = generate_dataset(1000)
    print(f"原始数据集形状: {df.shape}")
    
    # 数据清洗
    print("数据清洗中...")
    cleaned_df = clean_data(df)
    print(f"清洗后数据集形状: {cleaned_df.shape}")
    
    # 数据可视化分析
    print("进行数据可视化分析...")
    analyze_data(cleaned_df)
    
    # 划分特征和目标变量
    X = cleaned_df[['area', 'rooms', 'age']]
    y = cleaned_df['price']
    
    # 数据集划分：先划分为训练集和临时集，再将临时集划分为验证集和测试集
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 构建模型
    model = build_model(input_dim=X_train.shape[1])
    model.summary()
    
    # 定义早停策略，防止过拟合
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    # 学习率调度器
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # 训练模型
    print("开始训练模型...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_val_scaled, y_val),
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )
    
    # 绘制训练过程中的损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('训练与验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失 (MSE)')
    plt.legend()
    plt.savefig('plots/training_loss.png')
    plt.close()
    
    # 在测试集上评估模型
    print("在测试集上评估模型...")
    y_pred = model.predict(X_test_scaled).flatten()
    
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"测试集均方误差 (MSE): {mse:.2f}")
    print(f"测试集均方根误差 (RMSE): {rmse:.2f}")
    print(f"决定系数 (R²): {r2:.4f}")
    
    # 绘制预测值与实际值对比图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.title('实际价格 vs 预测价格')
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.savefig('plots/prediction_vs_actual.png')
    plt.close()
    
    # 绘制残差图
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测价格')
    plt.ylabel('残差')
    plt.title('残差图')
    plt.savefig('plots/residuals.png')
    plt.close()
    
    # 保存模型
    model.save('linear_regression_model.h5')
    print("模型已保存为 'linear_regression_model.h5'")
    
    return model, scaler, mse, rmse, r2

if __name__ == "__main__":
    # 设置随机种子，保证结果可复现
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 训练并评估模型
    model, scaler, mse, rmse, r2 = train_and_evaluate()
    
    # 模型应用示例
    print("\n模型应用示例:")
    # 假设一个新的房屋数据
    new_house = pd.DataFrame({
        'area': [120],    # 面积120平方米
        'rooms': [3],     # 3个房间
        'age': [5]        # 房龄5年
    })
    
    # 特征标准化
    new_house_scaled = scaler.transform(new_house)
    
    # 预测价格
    predicted_price = model.predict(new_house_scaled)[0][0]
    print(f"对于面积{new_house['area'][0]}平方米, {new_house['rooms'][0]}个房间, 房龄{new_house['age'][0]}年的房屋，")
    print(f"预测价格为: {predicted_price:.2f}元")