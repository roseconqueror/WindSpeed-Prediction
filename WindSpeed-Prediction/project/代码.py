import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import warnings
import optuna
import optuna.visualization.matplotlib as vis_mpl  # 引入Optuna绘图模块

warnings.filterwarnings('ignore')


# ==========================================
# 0. 全局随机种子 (保证结果 100% 可复现)
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# ==========================================
# 1. 核心运行模式与任务预设
# ==========================================
RUN_MODE = "EVAL"  # "EVAL": 极速读取模式；"TUNE": 重新启动 Optuna 调参

HISTORY_HOURS = 8
SEQ_LEN = HISTORY_HOURS * 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASKS = {
    "Single-step": 1,
    "Multi-step_A_1h": 6,
    "Multi-step_B_16h": 96
}


# ==========================================
# 2. 数据获取
# ==========================================
def load_and_merge_data():
    print("Downloading datasets from Hugging Face...")
    ds_10 = load_dataset("Antajitters/WindSpeed_10m", split="train").to_pandas()
    ds_50 = load_dataset("Antajitters/WindSpeed_50m", split="train").to_pandas()
    ds_100 = load_dataset("Antajitters/WindSpeed_100m", split="train").to_pandas()

    for df in [ds_10, ds_50, ds_100]:
        df.rename(columns={'Date & Time Stamp': 'Timestamp', 'SpeedAvg': 'Wind_Speed'}, inplace=True)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)

    ds_10 = ds_10.add_suffix('_10m')
    ds_50 = ds_50.add_suffix('_50m')
    ds_100 = ds_100.add_suffix('_100m')

    return pd.concat([ds_10, ds_50, ds_100], axis=1, join='inner')


class WindSpeedDataset(Dataset):
    def __init__(self, X, y, seq_len, pred_len):
        self.X, self.y = torch.FloatTensor(X), torch.FloatTensor(y)
        self.seq_len, self.pred_len = seq_len, pred_len

    def __len__(self):
        return len(self.X) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        return self.X[idx: idx + self.seq_len], self.y[idx + self.seq_len: idx + self.seq_len + self.pred_len]


# ==========================================
# 3. 严格防泄露的数据集划分与特征工程
# ==========================================
def prepare_data_pipeline(df, target_col, pred_len):
    print("\n" + "=" * 50)
    print("🛠️ 执行无泄露数据清洗与高级特征工程 (严格遵循数据集切分边界)")

    # 1. 物理切分数据集 (强制先切分，后计算)
    train_end = int(len(df) * 0.7)
    val_end = int(len(df) * 0.9)
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # 2. 缺失值处理: 向前填充，防止未来数据泄露
    for d in [train_df, val_df, test_df]:
        d.ffill(inplace=True)
        d.bfill(inplace=True)  # 仅处理头部极少量的初始空值

    # 3. IQR 异常值处理: 【仅】基于训练集计算阈值
    outlier_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if 'Direction' not in c]
    outlier_count = 0
    for col in outlier_cols:
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_count += ((train_df[col] < lower_bound) | (train_df[col] > upper_bound)).sum()

        # 将训练集的阈值规则应用到所有集合
        train_df[col] = np.clip(train_df[col], lower_bound, upper_bound)
        val_df[col] = np.clip(val_df[col], lower_bound, upper_bound)
        test_df[col] = np.clip(test_df[col], lower_bound, upper_bound)
    print(f"    [特征工程] 基于训练集IQR法则，平滑异常值数量: {outlier_count}")

    # 4. 风向特征拓扑变换: 消除 360 度到 0 度的跳跃
    dir_cols = [c for c in train_df.columns if 'Direction' in c]
    for col in dir_cols:
        for d in [train_df, val_df, test_df]:
            rad = d[col] * np.pi / 180.0
            d[f'{col}_sin'] = np.sin(rad)
            d[f'{col}_cos'] = np.cos(rad)
            d.drop(columns=[col], inplace=True)
    print(f"    [特征工程] 成功对所有 {len(dir_cols)} 个高度的风向进行 Sin/Cos 周期编码。")

    # 5. 时间周期编码与风切变
    for d in [train_df, val_df, test_df]:
        d['Hour_sin'] = np.sin(2 * np.pi * d.index.hour / 24.0)
        d['Hour_cos'] = np.cos(2 * np.pi * d.index.hour / 24.0)
        if 'Wind_Speed_100m' in d.columns and 'Wind_Speed_10m' in d.columns:
            d['Wind_Shear'] = d['Wind_Speed_100m'] - d['Wind_Speed_10m']

    # 6. 安全的动态滚动特征 (Rolling)
    window = 6
    train_df['Wind_Speed_10m_Rolling_Std'] = train_df['Wind_Speed_10m'].rolling(window=window).std().fillna(0)

    # 验证集: 取训练集最后5个真实值铺垫，计算后截断
    val_combined = pd.concat([train_df['Wind_Speed_10m'].iloc[-window + 1:], val_df['Wind_Speed_10m']])
    val_df['Wind_Speed_10m_Rolling_Std'] = val_combined.rolling(window=window).std().iloc[window - 1:].values

    # 测试集: 取验证集最后5个真实值铺垫，计算后截断
    test_combined = pd.concat([val_df['Wind_Speed_10m'].iloc[-window + 1:], test_df['Wind_Speed_10m']])
    test_df['Wind_Speed_10m_Rolling_Std'] = test_combined.rolling(window=window).std().iloc[window - 1:].values
    print("    [特征工程] 基于历史时间窗构造滚动标准差，数据泄露路径已彻底阻断。")
    print("=" * 50 + "\n")

    # 7. 全局归一化缩放 (Fit 仅限于训练集)
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    feature_cols = train_df.columns.tolist()

    X_train = scaler_X.fit_transform(train_df[feature_cols])
    y_train = scaler_y.fit_transform(train_df[[target_col]])

    X_val = scaler_X.transform(val_df[feature_cols])
    y_val = scaler_y.transform(val_df[[target_col]])

    X_test = scaler_X.transform(test_df[feature_cols])
    y_test = scaler_y.transform(test_df[[target_col]])

    train_dataset = WindSpeedDataset(X_train, y_train, SEQ_LEN, pred_len)
    val_dataset = WindSpeedDataset(X_val, y_val, SEQ_LEN, pred_len)
    test_dataset = WindSpeedDataset(X_test, y_test, SEQ_LEN, pred_len)

    return train_dataset, val_dataset, test_dataset, scaler_y, len(feature_cols)


# ==========================================
# 4. 模型定义
# ==========================================
class LinearModel(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, **kwargs):
        super().__init__()
        self.linear = nn.Linear(input_dim * seq_len, pred_len)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1)).unsqueeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, hidden_dim=64, num_layers=1, dropout=0.3, bidirectional=False):
        super().__init__()
        actual_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=actual_dropout,
                            bidirectional=bidirectional)
        dir_factor = 2 if bidirectional else 1
        self.fc1 = nn.Linear(hidden_dim * dir_factor, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc2(self.relu(self.fc1(out[:, -1, :]))).unsqueeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]


class TransformerModel(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, d_model=32, nhead=4, num_layers=1, dropout=0.3,
                 dim_feedforward=128):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(seq_len * d_model, pred_len)

    def forward(self, x):
        x = self.transformer(self.pos_encoder(self.embedding(x)))
        return self.fc(x.reshape(x.size(0), -1)).unsqueeze(-1)


class CNN1DModel(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, hidden_channels=64, dropout=0.3, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(input_dim, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear((hidden_channels * 2) * (seq_len // 4), pred_len)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x.permute(0, 2, 1)))))
        x = self.dropout(self.pool(self.relu(self.bn2(self.conv2(x)))))
        return self.fc(x.view(x.size(0), -1)).unsqueeze(-1)


# ==========================================
# 5. 训练机制与 Optuna 贝叶斯优化
# ==========================================
def train_model_core(model, train_loader, val_loader, lr, max_epochs, patience, weight_decay=1e-5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(max_epochs):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch.to(DEVICE)), y_batch.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_loss += criterion(model(X_batch.to(DEVICE)), y_batch.to(DEVICE)).item()
        val_loss /= len(val_loader)

        # 安全拦截机制：预防因极端学习率导致的NaN
        if math.isnan(val_loss):
            return best_weights, float('inf'), train_losses, val_losses

        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_weights, best_val_loss, train_losses, val_losses


def optuna_tune_model(ModelClass, train_dataset, val_dataset, num_features, pred_len, model_name, task_name):
    print(f"\n    [Optuna Tuning] {model_name} - Initiating Bayesian Optimization...")

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
        bs = trial.suggest_categorical("batch_size", [32, 64])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        model_kwargs = {}
        if model_name == "LSTM":
            model_kwargs['hidden_dim'] = trial.suggest_categorical("hidden_dim", [32, 64, 128])
            model_kwargs['num_layers'] = trial.suggest_int("num_layers", 1, 2)
            model_kwargs['dropout'] = trial.suggest_float("dropout", 0.1, 0.5)
            model_kwargs['bidirectional'] = trial.suggest_categorical("bidirectional", [True, False])
        elif model_name == "Transformer":
            d_model = trial.suggest_categorical("d_model", [16, 32, 64])
            valid_nheads = [h for h in [2, 4, 8] if d_model % h == 0]
            model_kwargs['d_model'] = d_model
            model_kwargs['nhead'] = trial.suggest_categorical("nhead", valid_nheads)
            model_kwargs['num_layers'] = 1
            model_kwargs['dropout'] = trial.suggest_float("dropout", 0.2, 0.6)
            dim_ff_mult = trial.suggest_categorical("dim_ff_mult", [2, 4])
            model_kwargs['dim_feedforward'] = d_model * dim_ff_mult
            lr = trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True)
        elif model_name == "CNN1D":
            model_kwargs['hidden_channels'] = trial.suggest_categorical("hidden_channels", [32, 64, 128])
            model_kwargs['dropout'] = trial.suggest_float("dropout", 0.2, 0.6)
            model_kwargs['kernel_size'] = trial.suggest_categorical("kernel_size", [3, 5, 7])

        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

        init_kwargs = copy.deepcopy(model_kwargs)
        if 'dim_ff_mult' in init_kwargs:
            init_kwargs.pop('dim_ff_mult')

        model = ModelClass(input_dim=num_features, seq_len=SEQ_LEN, pred_len=pred_len, **init_kwargs).to(DEVICE)
        _, val_loss, _, _ = train_model_core(model, train_loader, val_loader, lr, max_epochs=25, patience=5,
                                             weight_decay=weight_decay)
        return val_loss

    study = optuna.create_study(direction="minimize", study_name=f"{model_name}_{task_name}")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params
    print(f"    🌟 Best Params for {model_name}: {best_params}")

    # 📊 新增：保存 Optuna 高级分析图表
    try:
        fig_hist = vis_mpl.plot_optimization_history(study)
        fig_hist.figure.savefig(f"Optuna_History_{model_name}_{task_name}.png", bbox_inches='tight')
        plt.close(fig_hist.figure)

        fig_param = vis_mpl.plot_param_importances(study)
        fig_param.figure.savefig(f"Optuna_Param_Importances_{model_name}_{task_name}.png", bbox_inches='tight')
        plt.close(fig_param.figure)
    except Exception as e:
        print(f"    [WARNING] Optuna 图表生成跳过: {e}")

    final_lr = best_params.pop('learning_rate')
    final_bs = best_params.pop('batch_size')
    final_wd = best_params.pop('weight_decay')

    final_kwargs = copy.deepcopy(best_params)
    if model_name == "Transformer":
        final_kwargs['dim_feedforward'] = final_kwargs['d_model'] * final_kwargs.pop('dim_ff_mult')
        if 'num_layers' not in final_kwargs:
            final_kwargs['num_layers'] = 1

    train_loader = DataLoader(train_dataset, batch_size=final_bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=final_bs, shuffle=False)

    best_model = ModelClass(input_dim=num_features, seq_len=SEQ_LEN, pred_len=pred_len, **final_kwargs).to(DEVICE)
    best_weights, _, train_losses, val_losses = train_model_core(best_model, train_loader, val_loader, final_lr,
                                                                 max_epochs=60, patience=7, weight_decay=final_wd)

    # 📊 新增：保存模型训练收敛曲线图
    plot_learning_curve(train_losses, val_losses, model_name, task_name)

    best_model.load_state_dict(best_weights)
    best_params['learning_rate'] = final_lr
    best_params['batch_size'] = final_bs
    best_params['weight_decay'] = final_wd

    checkpoint = {'hyperparameters': best_params, 'state_dict': best_weights}
    torch.save(checkpoint, f"{model_name}_{task_name}_best.pth")

    return best_model


def get_model_from_checkpoint_or_train(ModelClass, model_name, task_name, num_features, pred_len, train_dataset,
                                       val_dataset):
    pth_filename = f"{model_name}_{task_name}_best.pth"

    if os.path.exists(pth_filename):
        print(f"    [INFO] 发现智能模型库 {pth_filename}，动态重构网络架构并注入预训练权重...")
        checkpoint = torch.load(pth_filename, map_location=DEVICE)
        model_params = checkpoint.get('hyperparameters', {})
        for k in ['learning_rate', 'batch_size', 'patience', 'max_epochs', 'weight_decay']:
            model_params.pop(k, None)

        if model_name == "Transformer" and 'dim_ff_mult' in model_params:
            model_params['dim_feedforward'] = model_params['d_model'] * model_params.pop('dim_ff_mult')

        model = ModelClass(input_dim=num_features, seq_len=SEQ_LEN, pred_len=pred_len, **model_params).to(DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    if model_name == "Linear":
        model = ModelClass(input_dim=num_features, seq_len=SEQ_LEN, pred_len=pred_len).to(DEVICE)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        best_weights, _, _, _ = train_model_core(model, train_loader, val_loader, lr=0.001, max_epochs=60, patience=7)

        torch.save({'hyperparameters': {}, 'state_dict': best_weights}, pth_filename)
        model.load_state_dict(best_weights)
        return model
    else:
        return optuna_tune_model(ModelClass, train_dataset, val_dataset, num_features, pred_len, model_name, task_name)


def evaluate_model(model, test_loader, scaler_y):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch.to(DEVICE)).cpu().numpy()
            y_batch = y_batch.numpy()
            for idx in range(outputs.shape[0]):
                predictions.extend(scaler_y.inverse_transform(outputs[idx]))
                actuals.extend(scaler_y.inverse_transform(y_batch[idx]))

    preds_flat = np.array(predictions).flatten()
    acts_flat = np.array(actuals).flatten()

    return {
        "MSE": mean_squared_error(acts_flat, preds_flat),
        "RMSE": np.sqrt(mean_squared_error(acts_flat, preds_flat)),
        "MAE": mean_absolute_error(acts_flat, preds_flat),
        "R2": r2_score(acts_flat, preds_flat)
    }, acts_flat, preds_flat


# ==========================================
# 6. 新增：深度可视化系统 (绘图与分析模块)
# ==========================================
def plot_learning_curve(train_losses, val_losses, model_name, task_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
    plt.title(f'Learning Curve - {model_name} ({task_name})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(f'learning_curve_{model_name}_{task_name}.png', bbox_inches='tight')
    plt.close()


def plot_residuals(actuals, models_preds, task_name):
    plt.figure(figsize=(12, 6))
    colors = ['red', 'blue', 'green', 'orange']
    for idx, (name, preds) in enumerate(models_preds.items()):
        residuals = np.array(preds) - np.array(actuals)
        sns.kdeplot(residuals, label=f'{name} Residuals', color=colors[idx], fill=True, alpha=0.3)
    plt.title(f'Prediction Residuals Density (Error Distribution) - {task_name}')
    plt.xlabel('Error (Predicted - Actual Wind Speed)')
    plt.ylabel('Density')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
    plt.legend()
    plt.savefig(f'predictions_residuals_{task_name}.png', bbox_inches='tight')
    plt.close()


def plot_data_analysis(df):
    plt.figure(figsize=(18, 7))
    plt.subplot(1, 2, 1)
    sns.histplot(df['Wind_Speed_10m'], kde=True, color='teal', label='10m Wind Speed', alpha=0.5, bins=50)
    sns.histplot(df['Wind_Speed_50m'], kde=True, color='orange', label='50m Wind Speed', alpha=0.4, bins=50)
    sns.histplot(df['Wind_Speed_100m'], kde=True, color='purple', label='100m Wind Speed', alpha=0.3, bins=50)
    plt.title('Wind Speed Distribution Across Different Heights')
    plt.xlabel('Wind Speed')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 2, 2)
    subset_cols = [c for c in df.columns if 'Speed' in c or 'Temperature' in c or 'Hour' in c]
    sns.heatmap(df[subset_cols].corr(), cmap='coolwarm', annot=True, fmt=".2f", annot_kws={"size": 7})
    plt.title('Multi-Height Feature Correlation Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('data_analysis_distribution_correlation.png', bbox_inches='tight')
    plt.close()


def plot_predictions(actuals, models_preds, task_name, subset_len=150):
    plt.figure(figsize=(15, 6))
    plt.plot(actuals[:subset_len], label='True Wind Speed', color='black', linewidth=2)
    colors = ['red', 'blue', 'green', 'orange']
    for idx, (name, preds) in enumerate(models_preds.items()):
        plt.plot(preds[:subset_len], label=f'{name}', color=colors[idx], alpha=0.7)
    plt.title(f'Wind Speed Prediction Comparison - {task_name}')
    plt.xlabel('Time Steps')
    plt.ylabel('Wind Speed')
    plt.legend()
    plt.savefig(f'predictions_comparison_{task_name}.png', bbox_inches='tight')
    plt.close()


# ==========================================
# 7. 主执行函数
# ==========================================
def main():
    df_raw = load_and_merge_data()
    # 严格遵循 7:2:1 划分，并在划分后安全进行处理
    train_dataset, val_dataset, test_dataset, scaler_y, num_features = prepare_data_pipeline(df_raw, 'Wind_Speed_10m',
                                                                                             pred_len=1)

    # 临时取一小部分用于生成分布图（不影响后续数据集）
    plot_data_analysis(df_raw)

    model_classes = {
        "Linear": LinearModel,
        "LSTM": LSTMModel,
        "Transformer": TransformerModel,
        "CNN1D": CNN1DModel
    }

    print(f"\n[{'*' * 10} 当前程序运行模式: {RUN_MODE} {'*' * 10}]\n")

    for task_name, pred_len in TASKS.items():
        print(f"\n{'=' * 60}")
        print(f"🚀 STARTING TASK: {task_name} (Predicting {pred_len} steps)")
        print(f"{'=' * 60}")

        # 针对不同的预测长度，重新获取安全截断的 Dataset
        train_ds, val_ds, test_ds, s_y, n_feat = prepare_data_pipeline(df_raw, 'Wind_Speed_10m', pred_len)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        results, all_preds = {}, {}

        for name, MClass in model_classes.items():
            if RUN_MODE == "TUNE" and name != "Linear":
                best_model = optuna_tune_model(MClass, train_ds, val_ds, n_feat, pred_len, name, task_name)
            else:
                best_model = get_model_from_checkpoint_or_train(MClass, name, task_name, n_feat, pred_len, train_ds,
                                                                val_ds)

            metrics, actuals, preds = evaluate_model(best_model, test_loader, s_y)
            results[name] = metrics
            all_preds[name] = preds

        # 📊 绘制时序对比图与残差密度图
        plot_predictions(actuals, all_preds, task_name)
        plot_residuals(actuals, all_preds, task_name)

        print(f"\n=== Final Performance for {task_name} ===")
        df_results = pd.DataFrame(results).T
        print(df_results)
        df_results.to_csv(f'Final_Metrics_{task_name}.csv')


if __name__ == "__main__":
    main()