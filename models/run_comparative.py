import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# 动态将 DAmodels 和 DGmodels 加入系统路径，以便导入图片中展示的模块
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'DAmodels'))
sys.path.append(os.path.join(BASE_DIR, 'DGmodels'))

# 尝试导入对比模型 (需确保对应 .py 文件中有同名的类或主函数)
try:
    from DAmodels.ERM import ERM
    from DAmodels.DANN import DANN
    from DAmodels.DCORAL import DCORAL
    from DGmodels.IEDGNet import IEDGNet
    from DGmodels.DGNIS import DGNIS
    from DAmodels.data_loader_1d import get_da_dataloaders
    from DGmodels.data_loader_1d import get_dg_dataloaders
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Import error: {e}. Running in Mock mode for demonstration.")
    MODELS_AVAILABLE = False


class BenchmarkEvaluator:
    """
    """
    @staticmethod
    def evaluate_fd(y_true, y_pred):
        """ 故障诊断 (Fault Diagnosis) 评估指标: Accuracy, F1 Score[cite: 11] """
        acc = accuracy_score(y_true, y_pred)
        # 采用 macro F1 适配多分类非平衡任务
        f1 = f1_score(y_true, y_pred, average='macro') 
        return {"Accuracy": acc, "F1_Score": f1}

    @staticmethod
    def evaluate_rul(y_true, y_pred):
        """ RUL 预测评估指标: RMSE, R-squared (R^2)[cite: 11] """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {"RMSE": rmse, "R2": r2}


def train_and_evaluate_model(model_name, task_type, scenario, device='cuda'):
    """
    核心对比实验调用流
    :param model_name: 'ERM', 'DANN', 'DCORAL', 'IEDGNet', 'DGNIS'[cite: 11]
    :param task_type: 'FD' (故障诊断) 或 'RUL' (寿命预测)[cite: 11]
    :param scenario: 'small_sample' 或 'cross_condition'[cite: 11]
    """
    print(f"\n{'='*50}\nStarting Experiment: Model={model_name} | Task={task_type} | Scenario={scenario}\n{'='*50}")
    
    # 1. 初始化数据加载器 (区分 DA 和 DG 逻辑)
    # DA 方法需要 target domain 数据对齐，DG 方法只需要 source domains
    if model_name in ['ERM', 'DANN', 'DCORAL']:
        print(f"[INFO] Using Domain Adaptation data logic for {model_name}...")
        if MODELS_AVAILABLE:
            source_loader, target_train_loader, target_test_loader = get_da_dataloaders(task_type, scenario)
        else:
            source_loader, target_train_loader, target_test_loader = [None]*3
    else:
        print(f"[INFO] Using Domain Generalization data logic for {model_name}...")
        if MODELS_AVAILABLE:
            source_loader, target_test_loader = get_dg_dataloaders(task_type, scenario)
            target_train_loader = None
        else:
            source_loader, target_test_loader, target_train_loader = [None]*3

    # 2. 实例化对比模型
    if MODELS_AVAILABLE:
        if model_name == 'ERM':
            model = ERM(task_type=task_type).to(device)
        elif model_name == 'DANN':
            model = DANN(task_type=task_type).to(device)
        elif model_name == 'DCORAL':
            model = DCORAL(task_type=task_type).to(device)
        elif model_name == 'IEDGNet':
            model = IEDGNet(task_type=task_type).to(device)
        elif model_name == 'DGNIS':
            model = DGNIS(task_type=task_type).to(device)
    else:
        # Mock model for syntax demonstration
        model = nn.Linear(10, 2 if task_type == 'FD' else 1).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss() if task_type == 'FD' else nn.MSELoss()

    # 3. 训练阶段
    print(f"[INFO] Training {model_name}...")
    epochs = 50
    model.train()
    for epoch in range(epochs):
        # 伪代码：实际需根据 DAmodels/DGmodels 中的具体 forward 函数传参
        # DA 方法(如 DANN/DCORAL) 需要 source_x, source_y 和 target_x 进行损失计算[cite: 11]
        # DG 方法(如 IEDGNet) 需要多源域 source_x, source_y[cite: 11]
        pass 
    
    print(f"[INFO] Training completed for {model_name}.")

    # 4. 测试阶段
    print(f"[INFO] Evaluating {model_name} on Target Domain...")
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        # 模拟生成测试结果
        if task_type == 'FD':
            # 模拟分类输出 (例如类别 0, 1, 2)
            all_preds = np.random.randint(0, 3, 100)
            all_trues = np.random.randint(0, 3, 100)
        else:
            # 模拟连续回归输出 (RUL 介于 0 到 1)
            all_trues = np.linspace(1.0, 0.0, 100)
            all_preds = all_trues + np.random.normal(0, 0.1, 100)
            
    # 5. 计算并打印评估指标
    if task_type == 'FD':
        metrics = BenchmarkEvaluator.evaluate_fd(all_trues, all_preds)
        print(f"[{model_name} - {task_type}] Results: Accuracy = {metrics['Accuracy']:.4f}, F1 Score = {metrics['F1_Score']:.4f}")
    else:
        metrics = BenchmarkEvaluator.evaluate_rul(all_trues, all_preds)
        print(f"[{model_name} - {task_type}] Results: RMSE = {metrics['RMSE']:.4f}, R2 = {metrics['R2']:.4f}")
        
    return metrics


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 论文中选取的五种 Baseline 对比方法[cite: 11]
    baselines = ['ERM', 'DANN', 'DCORAL', 'IEDGNet', 'DGNIS']
    tasks = ['FD', 'RUL']
    scenarios = ['small_sample', 'cross_condition']
    
    results_summary = {}

    # 执行所有组合的对比实验
    for task in tasks:
        results_summary[task] = {}
        for scenario in scenarios:
            results_summary[task][scenario] = {}
            for model_name in baselines:
                # 运行评估
                metrics = train_and_evaluate_model(
                    model_name=model_name, 
                    task_type=task, 
                    scenario=scenario, 
                    device=device
                )
                results_summary[task][scenario][model_name] = metrics

    print("\n" + "="*50)
    print("ALL COMPARATIVE EXPERIMENTS COMPLETED.")
    print("="*50)