import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class LinearRegressionModel:
    def __init__(self):
        self.w = 0.0
        self.b = 0.0
        self.loss_history = []
        self.w_history = []
        self.b_history = []
    
    def predict(self, X):
        """预测函数"""
        return self.w * X + self.b
    
    def compute_loss(self, X, y):
        """计算均方误差损失"""
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)
    
    def gradient_descent(self, X, y, learning_rate=0.01, epochs=1000):
        """梯度下降训练"""
        n = len(X)
        
        for epoch in range(epochs):
            # 预测值
            y_pred = self.predict(X)
            
            # 计算梯度
            dw = (-2/n) * np.sum(X * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)
            
            # 更新参数
            self.w -= learning_rate * dw
            self.b -= learning_rate * db
            
            # 记录历史值
            current_loss = self.compute_loss(X, y)
            self.loss_history.append(current_loss)
            self.w_history.append(self.w)
            self.b_history.append(self.b)
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {current_loss:.4f}, w: {self.w:.4f}, b: {self.b:.4f}')

def main():
    # 1. 用pandas读取train.csv数据
    try:
        # 如果文件存在，直接读取
        data = pd.read_csv('train.csv')
    except FileNotFoundError:
        # 如果文件不存在，创建示例数据
        print("未找到train.csv文件，创建示例数据...")
        np.random.seed(42)
        X = np.random.randn(100, 1) * 10
        y = 2.5 * X.flatten() + 1.5 + np.random.randn(100) * 2
        
        data = pd.DataFrame({
            'feature': X.flatten(),
            'target': y
        })
        data.to_csv('train.csv', index=False)
        print("已创建示例train.csv文件")
    
    print("数据概览：")
    print(data.head())
    print(f"\n数据形状：{data.shape}")
    
    # 准备数据
    X = data.iloc[:, 0].values  # 第一列作为特征
    y = data.iloc[:, 1].values  # 第二列作为目标
    
    # 2. 训练y=wx+b模型
    print("\n开始训练模型...")
    model = LinearRegressionModel()
    
    # 训练模型
    model.gradient_descent(X, y, learning_rate=0.001, epochs=2000)
    
    print(f"\n最终参数：w = {model.w:.4f}, b = {model.b:.4f}")
    print(f"最终损失：{model.loss_history[-1]:.4f}")
    
    # 3. 用matplotlib绘制关系图
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 图1: w和loss之间的关系
    ax1.plot(model.w_history, model.loss_history, 'b-', alpha=0.7)
    ax1.set_xlabel('权重 w')
    ax1.set_ylabel('损失 Loss')
    ax1.set_title('权重 w 与损失 Loss 的关系')
    ax1.grid(True, alpha=0.3)
    
    # 图2: b和loss之间的关系
    ax2.plot(model.b_history, model.loss_history, 'r-', alpha=0.7)
    ax2.set_xlabel('偏置 b')
    ax2.set_ylabel('损失 Loss')
    ax2.set_title('偏置 b 与损失 Loss 的关系')
    ax2.grid(True, alpha=0.3)

    ax3.plot(range(len(model.loss_history)), model.loss_history, 'g-')
    ax3.set_xlabel('迭代次数')
    ax3.set_ylabel('损失 Loss')
    ax3.set_title('损失函数下降曲线')
    ax3.grid(True, alpha=0.3)

    ax4.scatter(X, y, alpha=0.7, label='数据点')
    x_range = np.linspace(X.min(), X.max(), 100)
    y_pred = model.predict(x_range)
    ax4.plot(x_range, y_pred, 'r-', linewidth=2, label=f'回归线: y = {model.w:.2f}x + {model.b:.2f}')
    ax4.set_xlabel('特征 X')
    ax4.set_ylabel('目标 y')
    ax4.set_title('线性回归拟合结果')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_regression_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    results_df = pd.DataFrame({
        'epoch': range(len(model.loss_history)),
        'w': model.w_history,
        'b': model.b_history,
        'loss': model.loss_history
    })
    
    results_df.to_csv('training_results.csv', index=False)
    print("\n训练结果已保存到 training_results.csv")
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(f"""# 线性回归模型训练项目

### 项目描述
完成y=wx+b线性回归模型的训练，分析权重w、偏置b与损失函数loss之间的关系。

### 实现功能
1. 使用pandas读取train.csv数据
2. 实现梯度下降算法训练线性回归模型
3. 绘制w-loss、b-loss关系图
4. 可视化训练过程和结果

### 文件说明
- `linear_regression.py`: 主程序文件
- `train.csv`: 训练数据
- `training_results.csv`: 训练过程记录
- `linear_regression_results.png`: 结果可视化图

### 运行结果
- 最终权重 w: {model.w:.4f}
- 最终偏置 b: {model.b:.4f}
- 最终损失: {model.loss_history[-1]:.4f}
""")
    
    print("README.md文件已创建")

if __name__ == "__main__":
    main()

# git_config.py - Git提交配置脚本
import subprocess
import os

def setup_git():
    """设置Git仓库并提交代码"""
    
    # 检查是否在Git仓库中
    if not os.path.exists('.git'):
        # 初始化Git仓库
        subprocess.run(['git', 'init'])
        print("Git仓库初始化完成")
    
    # 添加所有文件
    subprocess.run(['git', 'add', '.'])
    
    # 提交代码（请将学号替换为实际学号）
    student_id = "3233022104"  # 请修改为你的实际学号
    commit_message = f"完成线性回归模型训练 - 学号: {student_id}"
    
    subprocess.run(['git', 'commit', '-m', commit_message])
    print(f"代码已提交，备注: {commit_message}")

if __name__ == "__main__":
    setup_git()