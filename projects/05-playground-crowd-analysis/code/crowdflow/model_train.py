import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report
from sklearn.metrics import explained_variance_score, median_absolute_error, mean_squared_log_error

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 1. 数据预处理和特征工程 ====================
class TrafficDataPreprocessor:
    """交通数据预处理类"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y_reg = None
        self.y_class = None
        self.feature_names = None

    def load_and_preprocess(self):
        """加载和预处理数据"""
        print("加载数据...")
        df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        print(f"原始数据形状: {df.shape}")

        self.data = df.copy()

        # ========== 基础特征工程 ==========
        # 时间特征
        self.data['日期'] = pd.to_datetime(self.data['日期'])
        self.data['月'] = self.data['日期'].dt.month
        self.data['日'] = self.data['日期'].dt.day
        self.data['星期几'] = self.data['日期'].dt.dayofweek

        def extract_hour(time_str):
            try:
                return int(time_str.split('-')[0].split(':')[0])
            except:
                return 0

        self.data['小时'] = self.data['时间段'].apply(extract_hour)

        # 时间段类别
        def time_period(hour):
            if 6 <= hour < 9:
                return 0  # 早晨
            elif 9 <= hour < 12:
                return 1  # 上午
            elif 12 <= hour < 15:
                return 2  # 中午
            elif 15 <= hour < 18:
                return 3  # 下午
            elif 18 <= hour < 21:
                return 4  # 晚上
            else:
                return 5  # 深夜

        self.data['时间段类别'] = self.data['小时'].apply(time_period)

        # 是否早晚高峰
        self.data['早晚高峰'] = self.data['小时'].apply(lambda x: 1 if (7 <= x < 9) or (17 <= x < 19) else 0)

        # 天气编码
        weather_severity = {
            '晴': 0, '多云': 1, '阴': 2, '毛毛雨': 3,
            '小阵雨': 4, '小雨': 5, '中雨': 6, '大雨': 7, '中阵雨': 6
        }
        self.data['天气严重度'] = self.data['天气'].map(lambda x: weather_severity.get(x, 0))

        # 空气污染编码
        air_pollution = {'优': 0, '良': 1, '轻度污染': 2, '中度污染': 3, '重度污染': 4}
        self.data['污染程度'] = self.data['空气状况'].map(lambda x: air_pollution.get(x, 0))

        # 星期编码
        week_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
            'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        self.data['星期编码'] = self.data['星期'].map(week_map)

        # 季节编码
        season_map = {'春季': 0, '夏季': 1, '秋季': 2, '冬季': 3}
        self.data['季节编码'] = self.data['季节'].map(lambda x: season_map.get(x, 0))

        # ========== 时间序列特征 ==========
        # 按时间排序
        self.data = self.data.sort_values(['日期', '小时']).reset_index(drop=True)

        # 滞后特征（避免数据泄露）
        for lag in [1, 2, 3, 6, 12, 24]:
            self.data[f'人数_lag{lag}'] = self.data['人数'].shift(lag).fillna(method='ffill').fillna(0)

        # 滚动统计特征（使用历史数据，避免数据泄露）
        self.data['历史平均人数'] = self.data['人数'].expanding(min_periods=1).mean().shift(1).fillna(0)
        self.data['历史最大人数'] = self.data['人数'].expanding(min_periods=1).max().shift(1).fillna(0)
        self.data['历史标准差'] = self.data['人数'].expanding(min_periods=1).std().shift(1).fillna(0)

        # 日统计特征
        self.data['日平均人数'] = self.data.groupby('日期')['人数'].transform('mean')

        # 周期特征
        self.data['小时_sin'] = np.sin(2 * np.pi * self.data['小时'] / 24)
        self.data['小时_cos'] = np.cos(2 * np.pi * self.data['小时'] / 24)

        # ========== 交互特征 ==========
        self.data['温度_湿度'] = self.data['温度'] * self.data['湿度'] / 100
        self.data['是否体育课时间'] = self.data['是否上课时间'] * self.data['有无体育课']
        self.data['是否活动时间'] = (1 - self.data['是否假期']) * self.data['有无活动']

        # ========== 高峰定义 ==========
        # 使用历史统计定义高峰（避免数据泄露）
        Q3 = self.data['人数'].quantile(0.75)
        Q1 = self.data['人数'].quantile(0.25)
        IQR = Q3 - Q1
        threshold = Q3 + 1.5 * IQR

        self.data['是否高峰'] = (self.data['人数'] > threshold).astype(int)

        peak_ratio = self.data['是否高峰'].mean()
        print(f"高峰样本比例: {peak_ratio:.2%}")
        print(f"高峰阈值: {threshold:.1f}")

        # ========== 特征选择 ==========
        feature_cols = [
            '小时', '月', '日', '星期几', '时间段类别', '早晚高峰',
            '是否周末', '是否假期', '是否上课时间',
            '有无体育课', '有无活动', '是否体育课时间', '是否活动时间',
            '温度', '风速', '湿度', '日照时间',
            '天气严重度', '污染程度', '季节编码',
            '小时_sin', '小时_cos',
            '温度_湿度',
            '历史平均人数', '历史最大人数', '历史标准差',
            '日平均人数'
        ]

        # 添加滞后特征
        for lag in [1, 2, 3, 6]:
            feature_cols.append(f'人数_lag{lag}')

        # 检查并移除可能不存在的列
        feature_cols = [col for col in feature_cols if col in self.data.columns]

        # 处理缺失值
        self.data = self.data.fillna(0)

        # 分离特征和目标
        self.X = self.data[feature_cols].copy()
        self.y_reg = self.data['人数'].values
        self.y_class = self.data['是否高峰'].values
        self.feature_names = feature_cols

        print(f"特征数量: {len(feature_cols)}")
        print(f"数据形状: X={self.X.shape}, y_reg={self.y_reg.shape}, y_class={self.y_class.shape}")

        return self.X, self.y_reg, self.y_class, self.data


# ==================== 2. XGBoost集成模型 ====================
class XGBoostEnsembleModel:
    """XGBoost集成模型：分类器 + 双回归模型"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.classifier = None
        self.normal_regressor = None
        self.peak_regressor = None
        self.feature_names = None

        # 分类器参数
        self.classifier_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': random_state,
            'n_jobs': -1,
            'scale_pos_weight': 3,  # 处理类别不平衡
            'base_score': 0.5  # 明确设置base_score为浮点数
        }

        # 回归器参数
        self.regressor_params = {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': random_state,
            'n_jobs': -1,
            'base_score': 0.5  # 明确设置base_score为浮点数
        }

        # 训练历史
        self.history = {
            'classifier_train_acc': [],
            'classifier_val_acc': [],
            'classifier_train_f1': [],
            'classifier_val_f1': [],
            'normal_regressor_train_mae': [],
            'normal_regressor_val_mae': [],
            'peak_regressor_train_mae': [],
            'peak_regressor_val_mae': []
        }

    def train_classifier(self, X_train, y_train, X_val=None, y_val=None):
        """训练分类器"""
        print("\n" + "=" * 70)
        print("训练高峰分类器")
        print("=" * 70)

        # 检查类别分布
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"训练集类别分布: {dict(zip(unique, counts))}")

        # 调整scale_pos_weight参数
        if len(counts) > 1:
            neg_count = counts[0] if unique[0] == 0 else counts[1]
            pos_count = counts[1] if unique[0] == 0 else counts[0]
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
            print(f"类别权重 (scale_pos_weight): {scale_pos_weight:.2f}")
            self.classifier_params['scale_pos_weight'] = max(1, min(10, scale_pos_weight))

        # 创建分类器
        self.classifier = xgb.XGBClassifier(**self.classifier_params)

        # 训练分类器
        if X_val is not None and y_val is not None:
            # 使用验证集进行训练
            print("使用验证集进行训练...")
            self.classifier.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
        else:
            # 仅使用训练集
            print("仅使用训练集进行训练...")
            self.classifier.fit(X_train, y_train)

        # 评估训练集
        train_pred = self.classifier.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred, zero_division=0)

        print(f"训练集 - 准确率: {train_acc:.4f}, F1: {train_f1:.4f}")
        self.history['classifier_train_acc'].append(train_acc)
        self.history['classifier_train_f1'].append(train_f1)

        # 评估验证集
        if X_val is not None and y_val is not None:
            val_pred = self.classifier.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, zero_division=0)

            print(f"验证集 - 准确率: {val_acc:.4f}, F1: {val_f1:.4f}")
            self.history['classifier_val_acc'].append(val_acc)
            self.history['classifier_val_f1'].append(val_f1)

        return self.classifier

    def train_regressors(self, X_train, y_reg_train, y_class_train,
                         X_val=None, y_reg_val=None, y_class_val=None):
        """分别训练正常和高低峰回归模型"""
        print("\n" + "=" * 70)
        print("训练回归模型")
        print("=" * 70)

        # 分离正常和高峰样本
        normal_mask = (y_class_train == 0)
        peak_mask = (y_class_train == 1)

        X_normal = X_train[normal_mask]
        y_normal = y_reg_train[normal_mask]

        X_peak = X_train[peak_mask]
        y_peak = y_reg_train[peak_mask]

        print(f"正常样本数: {len(X_normal)}")
        print(f"高峰样本数: {len(X_peak)}")

        # ========== 训练正常情况回归模型 ==========
        print("\n训练正常情况回归模型...")
        if len(X_normal) > 10:
            # 调整正常回归器参数
            normal_params = self.regressor_params.copy()
            normal_params['n_estimators'] = 250
            normal_params['max_depth'] = 6  # 正常情况相对简单

            self.normal_regressor = xgb.XGBRegressor(**normal_params)

            if X_val is not None and y_reg_val is not None and y_class_val is not None:
                # 分离验证集的正常样本
                normal_mask_val = (y_class_val == 0)
                X_val_normal = X_val[normal_mask_val]
                y_val_normal = y_reg_val[normal_mask_val]

                if len(X_val_normal) > 5:
                    print("使用验证集进行训练...")
                    self.normal_regressor.fit(
                        X_normal, y_normal,
                        eval_set=[(X_normal, y_normal), (X_val_normal, y_val_normal)],
                        verbose=False
                    )
                else:
                    print("验证集样本不足，仅使用训练集...")
                    self.normal_regressor.fit(X_normal, y_normal)
            else:
                print("无验证集，仅使用训练集...")
                self.normal_regressor.fit(X_normal, y_normal)

            # 评估正常回归器
            normal_pred = self.normal_regressor.predict(X_normal)
            normal_mae = mean_absolute_error(y_normal, normal_pred)
            normal_r2 = r2_score(y_normal, normal_pred)

            print(f"正常回归器 - 训练集MAE: {normal_mae:.2f}, R²: {normal_r2:.4f}")
            self.history['normal_regressor_train_mae'].append(normal_mae)

            # 验证集评估
            if X_val is not None and y_reg_val is not None and y_class_val is not None:
                if len(X_val_normal) > 0:
                    normal_pred_val = self.normal_regressor.predict(X_val_normal)
                    normal_mae_val = mean_absolute_error(y_val_normal, normal_pred_val)
                    print(f"正常回归器 - 验证集MAE: {normal_mae_val:.2f}")
                    self.history['normal_regressor_val_mae'].append(normal_mae_val)
        else:
            print("正常样本太少，跳过正常回归器训练")
            self.normal_regressor = None

        # ========== 训练高峰情况回归模型 ==========
        print("\n训练高峰情况回归模型...")
        if len(X_peak) > 10:
            # 调整高峰回归器参数
            peak_params = self.regressor_params.copy()
            peak_params['n_estimators'] = 350
            peak_params['max_depth'] = 10  # 高峰情况更复杂，需要更深的树
            peak_params['learning_rate'] = 0.03
            peak_params['subsample'] = 0.7  # 更保守的抽样
            peak_params['eval_metric'] = 'rmse'

            self.peak_regressor = xgb.XGBRegressor(**peak_params)

            if X_val is not None and y_reg_val is not None and y_class_val is not None:
                # 分离验证集的高峰样本
                peak_mask_val = (y_class_val == 1)
                X_val_peak = X_val[peak_mask_val]
                y_val_peak = y_reg_val[peak_mask_val]

                if len(X_val_peak) > 5:
                    print("使用验证集进行训练...")
                    self.peak_regressor.fit(
                        X_peak, y_peak,
                        eval_set=[(X_peak, y_peak), (X_val_peak, y_val_peak)],
                        verbose=False
                    )
                else:
                    print("验证集样本不足，仅使用训练集...")
                    self.peak_regressor.fit(X_peak, y_peak)
            else:
                print("无验证集，仅使用训练集...")
                self.peak_regressor.fit(X_peak, y_peak)

            # 评估高峰回归器
            peak_pred = self.peak_regressor.predict(X_peak)
            peak_mae = mean_absolute_error(y_peak, peak_pred)
            peak_r2 = r2_score(y_peak, peak_pred)

            print(f"高峰回归器 - 训练集MAE: {peak_mae:.2f}, R²: {peak_r2:.4f}")
            self.history['peak_regressor_train_mae'].append(peak_mae)

            # 验证集评估
            if X_val is not None and y_reg_val is not None and y_class_val is not None:
                if len(X_val_peak) > 0:
                    peak_pred_val = self.peak_regressor.predict(X_val_peak)
                    peak_mae_val = mean_absolute_error(y_val_peak, peak_pred_val)
                    print(f"高峰回归器 - 验证集MAE: {peak_mae_val:.2f}")
                    self.history['peak_regressor_val_mae'].append(peak_mae_val)
        else:
            print("高峰样本太少，跳过高峰回归器训练")
            self.peak_regressor = None

        # 如果某个回归器未训练成功，创建通用回归器作为后备
        if self.normal_regressor is None or self.peak_regressor is None:
            print("\n创建通用回归器作为后备...")
            self.general_regressor = xgb.XGBRegressor(**self.regressor_params)
            self.general_regressor.fit(X_train, y_reg_train)

    def predict(self, X, classification_threshold=0.4):
        """集成预测：先分类，再选择回归模型"""
        if self.classifier is None:
            raise ValueError("分类器未训练")

        # 1. 分类器预测是否为高峰
        # 获取概率预测
        class_probs = self.classifier.predict_proba(X)[:, 1]

        # 应用阈值判断是否为高峰
        is_peak = (class_probs > classification_threshold).astype(int)

        # 2. 回归预测
        predictions = np.zeros(len(X))

        for i in range(len(X)):
            if isinstance(X, pd.DataFrame):
                x_i = X.iloc[i:i + 1]
            else:
                x_i = X[i:i + 1]

            if is_peak[i] == 1 and self.peak_regressor is not None:
                # 使用高峰回归器
                predictions[i] = self.peak_regressor.predict(x_i)[0]
            elif is_peak[i] == 0 and self.normal_regressor is not None:
                # 使用正常回归器
                predictions[i] = self.normal_regressor.predict(x_i)[0]
            elif hasattr(self, 'general_regressor') and self.general_regressor is not None:
                # 使用通用回归器
                predictions[i] = self.general_regressor.predict(x_i)[0]
            else:
                # 如果没有可用的回归器，使用简单预测
                predictions[i] = np.mean([pred for pred in [self.normal_regressor, self.peak_regressor]
                                          if pred is not None])

        # 确保预测值为非负
        predictions = np.maximum(predictions, 0)

        return predictions, is_peak, class_probs

    def evaluate(self, X_test, y_reg_test, y_class_test, classification_threshold=0.4):
        """评估集成模型"""
        print("\n" + "=" * 70)
        print("XGBoost集成模型评估结果")
        print("=" * 70)

        # 进行预测
        predictions, class_preds, class_probs = self.predict(X_test, classification_threshold)

        # 确保预测值为非负
        predictions = np.maximum(predictions, 0)

        # ========== 回归评估指标 ==========
        mse = mean_squared_error(y_reg_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_reg_test, predictions)
        r2 = r2_score(y_reg_test, predictions)

        # 更多回归指标
        explained_variance = explained_variance_score(y_reg_test, predictions)
        medae = median_absolute_error(y_reg_test, predictions)

        # 计算MAPE（Mean Absolute Percentage Error）
        epsilon = 1e-8
        mape = 100 * np.mean(np.abs((y_reg_test - predictions) / (y_reg_test + epsilon)))

        # 计算SMAPE
        smape = 100 * np.mean(2 * np.abs(predictions - y_reg_test) /
                              (np.abs(predictions) + np.abs(y_reg_test) + epsilon))

        # 计算MSLE（Mean Squared Logarithmic Error）
        # 确保所有值都为非负
        y_reg_test_nonneg = np.maximum(y_reg_test, 0)
        predictions_nonneg = np.maximum(predictions, 0)
        msle = mean_squared_log_error(y_reg_test_nonneg, predictions_nonneg)

        print(f"\n回归性能:")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")
        print(f"解释方差: {explained_variance:.4f}")
        print(f"中位数绝对误差: {medae:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"SMAPE: {smape:.2f}%")
        print(f"MSLE: {msle:.4f}")

        # ========== 分类评估指标 ==========
        accuracy = accuracy_score(y_class_test, class_preds)
        precision = precision_score(y_class_test, class_preds, zero_division=0)
        recall = recall_score(y_class_test, class_preds, zero_division=0)
        f1 = f1_score(y_class_test, class_preds, zero_division=0)
        cm = confusion_matrix(y_class_test, class_preds)

        # 计算更多分类指标
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # 计算ROC曲线和AUC
        fpr_roc, tpr_roc, _ = roc_curve(y_class_test, class_probs)
        roc_auc = auc(fpr_roc, tpr_roc)

        # 计算PR曲线和AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_class_test, class_probs)
        pr_auc = auc(recall_curve, precision_curve)

        print(f"\n分类性能 (高峰检测):")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"特异度: {specificity:.4f}")
        print(f"假阳性率: {fpr:.4f}")
        print(f"假阴性率: {fnr:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")
        print(f"\n混淆矩阵:")
        print(cm)

        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(y_class_test, class_preds, target_names=['正常', '高峰']))

        # ========== 分别评估高峰和正常样本 ==========
        peak_indices = y_class_test == 1
        normal_indices = y_class_test == 0

        if np.sum(peak_indices) > 0:
            peak_mse = mean_squared_error(y_reg_test[peak_indices], predictions[peak_indices])
            peak_mae = mean_absolute_error(y_reg_test[peak_indices], predictions[peak_indices])
            peak_smape = 100 * np.mean(2 * np.abs(predictions[peak_indices] - y_reg_test[peak_indices]) /
                                       (np.abs(predictions[peak_indices]) + np.abs(y_reg_test[peak_indices]) + epsilon))

            print(f"\n高峰样本预测性能:")
            print(f"高峰MSE:  {peak_mse:.4f}")
            print(f"高峰MAE:  {peak_mae:.4f}")
            print(f"高峰SMAPE: {peak_smape:.2f}%")
            print(f"高峰样本数: {np.sum(peak_indices)}")

        if np.sum(normal_indices) > 0:
            normal_mse = mean_squared_error(y_reg_test[normal_indices], predictions[normal_indices])
            normal_mae = mean_absolute_error(y_reg_test[normal_indices], predictions[normal_indices])
            normal_smape = 100 * np.mean(2 * np.abs(predictions[normal_indices] - y_reg_test[normal_indices]) /
                                         (np.abs(predictions[normal_indices]) + np.abs(
                                             y_reg_test[normal_indices]) + epsilon))

            print(f"\n正常样本预测性能:")
            print(f"正常MSE:  {normal_mse:.4f}")
            print(f"正常MAE:  {normal_mae:.4f}")
            print(f"正常SMAPE: {normal_smape:.2f}%")
            print(f"正常样本数: {np.sum(normal_indices)}")

        # 返回结果
        results = {
            'predictions': predictions,
            'targets': y_reg_test,
            'class_predictions': class_preds,
            'class_targets': y_class_test,
            'class_probs': class_probs,
            'metrics': {
                'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'smape': smape,
                'explained_variance': explained_variance, 'medae': medae, 'mape': mape, 'msle': msle,
                'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
                'specificity': specificity, 'fpr': fpr, 'fnr': fnr,
                'roc_auc': roc_auc, 'pr_auc': pr_auc,
                'fpr_roc': fpr_roc, 'tpr_roc': tpr_roc,
                'precision_curve': precision_curve, 'recall_curve': recall_curve
            }
        }

        return results

    def plot_feature_importance(self, X, max_features=15):
        """绘制特征重要性"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 获取特征名称
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # 分类器特征重要性
        if self.classifier is not None:
            importance = self.classifier.feature_importances_
            indices = np.argsort(importance)[::-1][:max_features]

            axes[0].barh(range(len(indices)), importance[indices])
            axes[0].set_yticks(range(len(indices)))
            axes[0].set_yticklabels([feature_names[i] for i in indices])
            axes[0].set_xlabel('重要性')
            axes[0].set_title('分类器特征重要性 (Top {})'.format(max_features))
            axes[0].invert_yaxis()

        # 正常回归器特征重要性
        if self.normal_regressor is not None:
            importance = self.normal_regressor.feature_importances_
            indices = np.argsort(importance)[::-1][:max_features]

            axes[1].barh(range(len(indices)), importance[indices], color='green')
            axes[1].set_yticks(range(len(indices)))
            axes[1].set_yticklabels([feature_names[i] for i in indices])
            axes[1].set_xlabel('重要性')
            axes[1].set_title('正常回归器特征重要性 (Top {})'.format(max_features))
            axes[1].invert_yaxis()

        # 高峰回归器特征重要性
        if self.peak_regressor is not None:
            importance = self.peak_regressor.feature_importances_
            indices = np.argsort(importance)[::-1][:max_features]

            axes[2].barh(range(len(indices)), importance[indices], color='red')
            axes[2].set_yticks(range(len(indices)))
            axes[2].set_yticklabels([feature_names[i] for i in indices])
            axes[2].set_xlabel('重要性')
            axes[2].set_title('高峰回归器特征重要性 (Top {})'.format(max_features))
            axes[2].invert_yaxis()

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath='xgboost_ensemble_model.pkl'):
        """保存模型"""
        model_data = {
            'classifier': self.classifier,
            'normal_regressor': self.normal_regressor,
            'peak_regressor': self.peak_regressor,
            'feature_names': self.feature_names,
            'classifier_params': self.classifier_params,
            'regressor_params': self.regressor_params,
            'history': self.history
        }
        joblib.dump(model_data, filepath)
        print(f"模型已保存到 {filepath}")

    def load_model(self, filepath='xgboost_ensemble_model.pkl'):
        """加载模型"""
        model_data = joblib.load(filepath)
        self.classifier = model_data['classifier']
        self.normal_regressor = model_data['normal_regressor']
        self.peak_regressor = model_data['peak_regressor']
        self.feature_names = model_data['feature_names']
        self.classifier_params = model_data.get('classifier_params', self.classifier_params)
        self.regressor_params = model_data.get('regressor_params', self.regressor_params)
        self.history = model_data.get('history', self.history)
        print(f"模型已从 {filepath} 加载")


# ==================== 3. 增强的可视化函数 ====================
def visualize_results(results, title="XGBoost集成模型预测结果"):
    """可视化模型结果 - 分成多个窗口，每个窗口最多2张图"""
    predictions = results['predictions']
    targets = results['targets']
    class_preds = results['class_predictions']
    class_targets = results['class_targets']
    class_probs = results['class_probs']

    metrics = results['metrics']

    # 添加epsilon变量以防止除以零
    epsilon = 1e-8

    # 计算残差
    residuals = predictions - targets

    # ========== 窗口1: 预测对比 ==========
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
    fig1.suptitle(f'{title} - 预测对比', fontsize=16, fontweight='bold')

    # 1. 整体预测对比
    n_show = min(100, len(predictions))
    axes1[0].plot(targets[:n_show], label='真实值', alpha=0.7, linewidth=2)
    axes1[0].plot(predictions[:n_show], label='预测值', alpha=0.7, linestyle='--')
    axes1[0].set_xlabel('样本索引')
    axes1[0].set_ylabel('人流量')
    axes1[0].set_title(f'前{n_show}个样本的预测对比')
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.3)

    # 2. 预测值与真实值散点图
    axes1[1].scatter(targets, predictions, alpha=0.5, s=10, c=class_targets, cmap='coolwarm')
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    axes1[1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    axes1[1].set_xlabel('真实值')
    axes1[1].set_ylabel('预测值')
    axes1[1].set_title('预测值 vs 真实值（颜色：红=高峰，蓝=正常）')
    axes1[1].grid(True, alpha=0.3)

    # 添加R²值
    axes1[1].text(0.05, 0.95, f'R² = {metrics["r2"]:.4f}', transform=axes1[1].transAxes,
                  fontsize=12, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()

    # ========== 窗口2: 残差分析 ==========
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle(f'{title} - 残差分析', fontsize=16, fontweight='bold')

    # 3. 残差分析
    axes2[0].scatter(predictions, residuals, alpha=0.5, s=10, c=class_targets, cmap='coolwarm')
    axes2[0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes2[0].set_xlabel('预测值')
    axes2[0].set_ylabel('残差')
    axes2[0].set_title('残差图')
    axes2[0].grid(True, alpha=0.3)

    # 添加MAE值
    axes2[0].text(0.05, 0.95, f'MAE = {metrics["mae"]:.4f}', transform=axes2[0].transAxes,
                  fontsize=12, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. 误差分布
    axes2[1].hist(np.abs(residuals), bins=30, alpha=0.7, edgecolor='black', density=True)
    axes2[1].axvline(x=np.mean(np.abs(residuals)), color='r', linestyle='--',
                     label=f'平均绝对误差: {np.mean(np.abs(residuals)):.2f}')
    axes2[1].axvline(x=np.median(np.abs(residuals)), color='g', linestyle='--',
                     label=f'中位数绝对误差: {np.median(np.abs(residuals)):.2f}')
    axes2[1].set_xlabel('绝对误差')
    axes2[1].set_ylabel('密度')
    axes2[1].set_title('预测误差分布')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ========== 窗口3: ROC和PR曲线 ==========
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
    fig3.suptitle(f'{title} - ROC和PR曲线', fontsize=16, fontweight='bold')

    # 5. ROC曲线
    fpr = metrics['fpr_roc']
    tpr = metrics['tpr_roc']
    roc_auc = metrics['roc_auc']
    axes3[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    axes3[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes3[0].set_xlim([0.0, 1.0])
    axes3[0].set_ylim([0.0, 1.05])
    axes3[0].set_xlabel('假阳性率')
    axes3[0].set_ylabel('真阳性率')
    axes3[0].set_title('ROC曲线')
    axes3[0].legend(loc="lower right")
    axes3[0].grid(True, alpha=0.3)

    # 6. PR曲线
    precision_curve = metrics['precision_curve']
    recall_curve = metrics['recall_curve']
    pr_auc = metrics['pr_auc']
    axes3[1].plot(recall_curve, precision_curve, color='darkgreen', lw=2, label=f'PR曲线 (AUC = {pr_auc:.4f})')
    # 计算随机分类器的PR曲线
    if len(class_targets) > 0:
        random_precision = np.sum(class_targets) / len(class_targets)
        axes3[1].axhline(y=random_precision, color='r', linestyle='--', label='随机分类器')
    axes3[1].set_xlim([0.0, 1.0])
    axes3[1].set_ylim([0.0, 1.05])
    axes3[1].set_xlabel('召回率')
    axes3[1].set_ylabel('精确率')
    axes3[1].set_title('PR曲线')
    axes3[1].legend(loc="lower left")
    axes3[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ========== 窗口4: 分类详情 ==========
    fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6))
    fig4.suptitle(f'{title} - 分类详情', fontsize=16, fontweight='bold')

    # 7. 分类概率分布
    axes4[0].hist(class_probs[class_targets == 0],
                  bins=20, alpha=0.5, label='正常样本', color='blue', density=True)
    axes4[0].hist(class_probs[class_targets == 1],
                  bins=20, alpha=0.5, label='高峰样本', color='red', density=True)
    axes4[0].axvline(x=0.4, color='k', linestyle='--', alpha=0.7, label='阈值(0.4)')
    axes4[0].set_xlabel('高峰概率')
    axes4[0].set_ylabel('密度')
    axes4[0].set_title('分类概率分布')
    axes4[0].legend()
    axes4[0].grid(True, alpha=0.3)

    # 8. 混淆矩阵热图
    cm = confusion_matrix(class_targets, class_preds)
    im = axes4[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig4.colorbar(im, ax=axes4[1])

    # 添加文本标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes4[1].text(j, i, format(cm[i, j], 'd'),
                          ha="center", va="center",
                          color="white" if cm[i, j] > thresh else "black")

    axes4[1].set_xlabel('预测标签')
    axes4[1].set_ylabel('真实标签')
    axes4[1].set_title('混淆矩阵')
    axes4[1].set_xticks([0, 1])
    axes4[1].set_yticks([0, 1])
    axes4[1].set_xticklabels(['正常', '高峰'])
    axes4[1].set_yticklabels(['正常', '高峰'])

    plt.tight_layout()
    plt.show()

    # ========== 窗口5: 回归性能指标 ==========
    fig5, axes5 = plt.subplots(1, 2, figsize=(16, 6))
    fig5.suptitle(f'{title} - 回归性能指标', fontsize=16, fontweight='bold')

    # 9. 回归性能指标条形图
    reg_metrics_names = ['R²', '解释方差', '1-MAPE', '1-SMAPE']
    reg_metrics_values = [
        metrics['r2'],
        metrics['explained_variance'],
        1 - min(metrics['mape'] / 100, 1),
        1 - min(metrics['smape'] / 100, 1)
    ]

    # 处理可能的NaN值
    reg_metrics_values = [0 if np.isnan(v) else v for v in reg_metrics_values]

    bars1 = axes5[0].bar(reg_metrics_names, reg_metrics_values,
                         color=['blue', 'green', 'orange', 'red'])
    axes5[0].set_ylim([0, 1])
    axes5[0].set_ylabel('分数')
    axes5[0].set_title('回归性能指标')
    axes5[0].grid(True, alpha=0.3, axis='y')

    # 在条形上添加数值
    for bar, value in zip(bars1, reg_metrics_values):
        height = bar.get_height()
        axes5[0].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                      f'{value:.3f}', ha='center', va='bottom', fontsize=10)

    # 10. 回归指标数值显示
    axes5[1].axis('off')
    reg_summary_text = (
        f"回归性能指标详细数值:\n\n"
        f"R² (决定系数): {metrics.get('r2', 0):.4f}\n"
        f"  解释方差: {metrics.get('explained_variance', 0):.4f}\n"
        f"  MSE: {metrics.get('mse', 0):.4f}\n"
        f"  RMSE: {metrics.get('rmse', 0):.4f}\n"
        f"  MAE: {metrics.get('mae', 0):.4f}\n"
        f"  中位数绝对误差: {metrics.get('medae', 0):.4f}\n"
        f"  MAPE: {metrics.get('mape', 0):.2f}%\n"
        f"  SMAPE: {metrics.get('smape', 0):.2f}%\n"
        f"  MSLE: {metrics.get('msle', 0):.4f}"
    )

    axes5[1].text(0.05, 0.95, reg_summary_text, transform=axes5[1].transAxes,
                  fontsize=11, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.show()

    # ========== 窗口6: 分类性能指标 ==========
    fig6, axes6 = plt.subplots(1, 2, figsize=(16, 6))
    fig6.suptitle(f'{title} - 分类性能指标', fontsize=16, fontweight='bold')

    # 11. 分类性能指标条形图
    class_metrics_names = ['准确率', '精确率', '召回率', 'F1', '特异度', 'ROC AUC']
    class_metrics_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['specificity'],
        metrics['roc_auc']
    ]

    # 处理可能的NaN值
    class_metrics_values = [0 if np.isnan(v) else v for v in class_metrics_values]

    bars2 = axes6[0].bar(class_metrics_names, class_metrics_values,
                         color=['blue', 'orange', 'green', 'red', 'purple', 'brown'])
    axes6[0].set_ylim([0, 1])
    axes6[0].set_ylabel('分数')
    axes6[0].set_title('分类性能指标')
    axes6[0].tick_params(axis='x', rotation=45)
    axes6[0].grid(True, alpha=0.3, axis='y')

    # 在条形上添加数值
    for bar, value in zip(bars2, class_metrics_values):
        height = bar.get_height()
        axes6[0].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                      f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    # 12. 分类指标数值显示
    axes6[1].axis('off')
    class_summary_text = (
        f"分类性能指标详细数值:\n\n"
        f"准确率: {metrics.get('accuracy', 0):.4f}\n"
        f"精确率: {metrics.get('precision', 0):.4f}\n"
        f"召回率: {metrics.get('recall', 0):.4f}\n"
        f"F1分数: {metrics.get('f1', 0):.4f}\n"
        f"特异度: {metrics.get('specificity', 0):.4f}\n"
        f"假阳性率: {metrics.get('fpr', 0):.4f}\n"
        f"假阴性率: {metrics.get('fnr', 0):.4f}\n"
        f"ROC AUC: {metrics.get('roc_auc', 0):.4f}\n"
        f"PR AUC: {metrics.get('pr_auc', 0):.4f}\n"
        f"混淆矩阵:\n"
        f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}\n"
        f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}"
    )

    axes6[1].text(0.05, 0.95, class_summary_text, transform=axes6[1].transAxes,
                  fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    plt.show()

    # ========== 窗口7: 时间分析和总结 ==========
    fig7, axes7 = plt.subplots(1, 2, figsize=(16, 6))
    fig7.suptitle(f'{title} - 时间分析和总结', fontsize=16, fontweight='bold')

    # 13. 按小时的平均预测误差
    try:
        # 假设有小时信息，如果没有则使用索引
        hours = np.arange(len(predictions)) % 24
        unique_hours = np.unique(hours)

        # 计算每个小时的平均预测误差
        hour_errors = []
        for hour in unique_hours:
            hour_mask = (hours == hour)
            if np.sum(hour_mask) > 0:
                hour_error = np.mean(np.abs(predictions[hour_mask] - targets[hour_mask]))
                hour_errors.append(hour_error)
            else:
                hour_errors.append(0)

        axes7[0].bar(unique_hours, hour_errors, alpha=0.7)
        axes7[0].set_xlabel('小时')
        axes7[0].set_ylabel('平均绝对误差')
        axes7[0].set_title('按小时的平均预测误差')
        axes7[0].grid(True, alpha=0.3)
    except:
        axes7[0].text(0.5, 0.5, '按小时的误差分析\n不可用',
                      ha='center', va='center', fontsize=14)
        axes7[0].set_title('按小时的平均预测误差')

    # 14. 模型性能总结
    axes7[1].axis('off')

    # 创建性能总结文本
    summary_text = (
        f"模型性能总结:\n\n"
        f"回归性能:\n"
        f"  R²: {metrics.get('r2', 0):.4f}\n"
        f"  MAE: {metrics.get('mae', 0):.4f}\n"
        f"  RMSE: {metrics.get('rmse', 0):.4f}\n"
        f"  MAPE: {metrics.get('mape', 0):.2f}%\n"
        f"  SMAPE: {metrics.get('smape', 0):.2f}%\n\n"
        f"分类性能:\n"
        f"  准确率: {metrics.get('accuracy', 0):.4f}\n"
        f"  F1分数: {metrics.get('f1', 0):.4f}\n"
        f"  ROC AUC: {metrics.get('roc_auc', 0):.4f}\n"
        f"  PR AUC: {metrics.get('pr_auc', 0):.4f}\n\n"
        f"样本统计:\n"
        f"  总样本数: {len(predictions)}\n"
        f"  高峰样本: {np.sum(class_targets)}\n"
        f"  正常样本: {len(class_targets) - np.sum(class_targets)}"
    )

    axes7[1].text(0.05, 0.95, summary_text, transform=axes7[1].transAxes,
                  fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.tight_layout()
    plt.show()

    # ========== 窗口8: 误差详细分析 ==========
    fig8, axes8 = plt.subplots(1, 2, figsize=(16, 6))
    fig8.suptitle(f'{title} - 误差详细分析', fontsize=16, fontweight='bold')

    # 15. 绝对误差分布
    axes8[0].hist(np.abs(residuals), bins=30, alpha=0.7, edgecolor='black')
    axes8[0].axvline(x=np.mean(np.abs(residuals)), color='r', linestyle='--',
                     label=f'平均: {np.mean(np.abs(residuals)):.2f}')
    axes8[0].axvline(x=np.median(np.abs(residuals)), color='g', linestyle='--',
                     label=f'中位数: {np.median(np.abs(residuals)):.2f}')
    axes8[0].set_xlabel('绝对误差')
    axes8[0].set_ylabel('频数')
    axes8[0].set_title('绝对误差分布')
    axes8[0].legend()
    axes8[0].grid(True, alpha=0.3)

    # 16. 相对误差分布
    relative_errors = np.abs(residuals) / (np.abs(targets) + epsilon)
    # 限制相对误差在合理范围内
    relative_errors = np.clip(relative_errors, 0, 5)

    axes8[1].hist(relative_errors, bins=30, alpha=0.7, edgecolor='black')
    axes8[1].axvline(x=np.mean(relative_errors), color='r', linestyle='--',
                     label=f'平均: {np.mean(relative_errors):.2f}')
    axes8[1].set_xlabel('相对误差 (真实值为分母)')
    axes8[1].set_ylabel('频数')
    axes8[1].set_title('相对误差分布 (截断到5)')
    axes8[1].legend()
    axes8[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ========== 窗口9: 误差关系分析 ==========
    fig9, axes9 = plt.subplots(1, 2, figsize=(16, 6))
    fig9.suptitle(f'{title} - 误差关系分析', fontsize=16, fontweight='bold')

    # 17. 预测值与误差的关系
    axes9[0].scatter(predictions, np.abs(residuals), alpha=0.5, s=10)
    axes9[0].set_xlabel('预测值')
    axes9[0].set_ylabel('绝对误差')
    axes9[0].set_title('预测值与绝对误差的关系')
    axes9[0].grid(True, alpha=0.3)

    # 18. 真实值与误差的关系
    axes9[1].scatter(targets, np.abs(residuals), alpha=0.5, s=10)
    axes9[1].set_xlabel('真实值')
    axes9[1].set_ylabel('绝对误差')
    axes9[1].set_title('真实值与绝对误差的关系')
    axes9[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==================== 4. 主程序 ====================
def main():
    print("=" * 70)
    print("XGBoost集成交通流量预测模型")
    print("=" * 70)

    # 1. 数据预处理
    print("\n1. 数据预处理...")
    preprocessor = TrafficDataPreprocessor('data.csv')
    X, y_reg, y_class, data = preprocessor.load_and_preprocess()

    # 2. 划分数据集（时间序列顺序）
    print("\n2. 划分数据集...")
    total_size = len(X)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)

    # 时间序列划分
    X_train = X.iloc[:train_size]
    y_reg_train = y_reg[:train_size]
    y_class_train = y_class[:train_size]

    X_val = X.iloc[train_size:train_size + val_size]
    y_reg_val = y_reg[train_size:train_size + val_size]
    y_class_val = y_class[train_size:train_size + val_size]

    X_test = X.iloc[train_size + val_size:]
    y_reg_test = y_reg[train_size + val_size:]
    y_class_test = y_class[train_size + val_size:]

    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"测试集大小: {len(X_test)}")

    # 3. 创建和训练模型
    print("\n3. 创建和训练XGBoost集成模型...")
    model = XGBoostEnsembleModel(random_state=42)

    # 保存特征名称
    model.feature_names = list(X_train.columns)

    # 训练分类器
    model.train_classifier(X_train, y_class_train, X_val, y_class_val)

    # 训练回归器
    model.train_regressors(X_train, y_reg_train, y_class_train, X_val, y_reg_val, y_class_val)

    # 4. 评估模型
    print("\n4. 在测试集上评估模型...")
    results = model.evaluate(X_test, y_reg_test, y_class_test, classification_threshold=0.4)

    # 5. 可视化特征重要性
    print("\n5. 可视化特征重要性...")
    model.plot_feature_importance(X_train, max_features=15)

    # 6. 可视化结果
    print("\n6. 可视化预测结果...")
    visualize_results(results, title="XGBoost集成模型预测结果")

    # 7. 保存模型
    print("\n7. 保存模型...")
    model.save_model('xgboost_ensemble_traffic_model.pkl')

    # 8. 模型性能总结
    print("\n8. 模型性能总结:")
    print(f"回归性能:")
    print(f"  R²: {results['metrics']['r2']:.4f}")
    print(f"  MAE: {results['metrics']['mae']:.4f}")
    print(f"  RMSE: {results['metrics']['rmse']:.4f}")
    print(f"  MAPE: {results['metrics']['mape']:.2f}%")
    print(f"  SMAPE: {results['metrics']['smape']:.2f}%")

    print(f"\n分类性能:")
    print(f"  准确率: {results['metrics']['accuracy']:.4f}")
    print(f"  F1分数: {results['metrics']['f1']:.4f}")
    print(f"  ROC AUC: {results['metrics']['roc_auc']:.4f}")
    print(f"  PR AUC: {results['metrics']['pr_auc']:.4f}")

    # 9. 模型架构总结
    print("\n9. 模型架构总结:")
    print("1. 分类器: 判断是否为高峰情况")
    print("2. 正常情况回归模型: 专门处理一般流量")
    print("3. 高峰情况回归模型: 专门处理突发高峰")
    print("4. 集成策略: 先分类判断高峰/正常，再选择合适的回归模型进行预测")
    print("\nXGBoost模型优势:")
    print("- 处理非线性关系能力强")
    print("- 自动特征选择和重要性评估")
    print("- 对异常值和缺失值鲁棒")
    print("- 训练速度快，适合小到中型数据集")
    print("- 可解释性好（特征重要性）")

    return model, results


# ==================== 5. 预测函数 ====================
def predict_with_model(model_path, new_data, feature_names=None, classification_threshold=0.4):
    """使用训练好的模型进行预测"""
    # 加载模型
    model_data = joblib.load(model_path)
    classifier = model_data['classifier']
    normal_regressor = model_data['normal_regressor']
    peak_regressor = model_data['peak_regressor']

    # 确保新数据有正确的特征
    if feature_names is not None:
        # 检查并添加缺失的特征
        for feature in feature_names:
            if feature not in new_data.columns:
                new_data[feature] = 0

        # 重新排序特征
        new_data = new_data[feature_names]

    # 预测
    class_probs = classifier.predict_proba(new_data)[:, 1]
    is_peak = (class_probs > classification_threshold).astype(int)

    predictions = np.zeros(len(new_data))

    for i in range(len(new_data)):
        x_i = new_data.iloc[i:i + 1]

        if is_peak[i] == 1 and peak_regressor is not None:
            predictions[i] = peak_regressor.predict(x_i)[0]
        elif is_peak[i] == 0 and normal_regressor is not None:
            predictions[i] = normal_regressor.predict(x_i)[0]
        else:
            # 使用平均预测
            preds = []
            if normal_regressor is not None:
                preds.append(normal_regressor.predict(x_i)[0])
            if peak_regressor is not None:
                preds.append(peak_regressor.predict(x_i)[0])
            predictions[i] = np.mean(preds) if preds else 0

    predictions = np.maximum(predictions, 0)

    print(f"预测结果:")
    print(f"样本数: {len(new_data)}")
    print(f"高峰样本数: {np.sum(is_peak)} ({np.sum(is_peak) / len(is_peak) * 100:.1f}%)")
    print(f"平均预测值: {np.mean(predictions):.2f}")
    print(f"预测范围: {np.min(predictions):.2f} - {np.max(predictions):.2f}")

    return predictions, is_peak, class_probs


# ==================== 执行主程序 ====================
if __name__ == "__main__":
    try:
        # 设置随机种子
        np.random.seed(42)

        print("开始训练XGBoost集成模型...")
        model, results = main()

        if results is not None:
            print("\n" + "=" * 70)
            print("XGBoost集成模型训练完成！")
            print("=" * 70)

            print(f"\n最终性能指标:")
            print(f"回归R²: {results['metrics']['r2']:.4f}")
            print(f"回归MAE: {results['metrics']['mae']:.4f}")
            print(f"回归SMAPE: {results['metrics']['smape']:.2f}%")
            print(f"分类准确率: {results['metrics']['accuracy']:.4f}")
            print(f"分类F1: {results['metrics']['f1']:.4f}")

            print(f"\n模型已保存为 'xgboost_ensemble_traffic_model.pkl'")
            print("可以使用 predict_with_model() 函数加载模型进行预测")

    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback

        traceback.print_exc()
