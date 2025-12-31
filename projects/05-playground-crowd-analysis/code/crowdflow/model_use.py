import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class TrafficDataPreprocessor:
    """交通数据预处理类（简化版，用于应用）"""

    @staticmethod
    def preprocess_input_data(df):
        """预处理输入数据（从原始数据生成特征）"""
        data = df.copy()

        # ========== 基础特征工程 ==========
        # 时间特征
        if '日期' in data.columns:
            data['日期'] = pd.to_datetime(data['日期'])
            data['月'] = data['日期'].dt.month
            data['日'] = data['日期'].dt.day
            data['星期几'] = data['日期'].dt.dayofweek

        # 提取小时
        def extract_hour(time_str):
            try:
                if pd.isna(time_str):
                    return 0
                return int(str(time_str).split('-')[0].split(':')[0])
            except:
                return 0

        if '时间段' in data.columns:
            data['小时'] = data['时间段'].apply(extract_hour)
        else:
            data['小时'] = 0

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

        data['时间段类别'] = data['小时'].apply(time_period)

        # 是否早晚高峰
        data['早晚高峰'] = data['小时'].apply(lambda x: 1 if (7 <= x < 9) or (17 <= x < 19) else 0)

        # 天气编码
        weather_severity = {
            '晴': 0, '多云': 1, '阴': 2, '毛毛雨': 3,
            '小阵雨': 4, '小雨': 5, '中雨': 6, '大雨': 7, '中阵雨': 6
        }
        if '天气' in data.columns:
            data['天气严重度'] = data['天气'].map(lambda x: weather_severity.get(x, 0))
        else:
            data['天气严重度'] = 0

        # 空气污染编码
        air_pollution = {'优': 0, '良': 1, '轻度污染': 2, '中度污染': 3, '重度污染': 4}
        if '空气状况' in data.columns:
            data['污染程度'] = data['空气状况'].map(lambda x: air_pollution.get(x, 0))
        else:
            data['污染程度'] = 0

        # 星期编码
        week_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
            'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        if '星期' in data.columns:
            data['星期编码'] = data['星期'].map(week_map)
        else:
            data['星期编码'] = 0

        # 季节编码
        season_map = {'春季': 0, '夏季': 1, '秋季': 2, '冬季': 3}
        if '季节' in data.columns:
            data['季节编码'] = data['季节'].map(lambda x: season_map.get(x, 0))
        else:
            data['季节编码'] = 0

        # 周期特征
        data['小时_sin'] = np.sin(2 * np.pi * data['小时'] / 24)
        data['小时_cos'] = np.cos(2 * np.pi * data['小时'] / 24)

        # 交互特征
        if '温度' in data.columns and '湿度' in data.columns:
            data['温度_湿度'] = data['温度'] * data['湿度'] / 100
        else:
            data['温度_湿度'] = 0

        if '是否上课时间' in data.columns and '有无体育课' in data.columns:
            data['是否体育课时间'] = data['是否上课时间'] * data['有无体育课']
        else:
            data['是否体育课时间'] = 0

        if '是否假期' in data.columns and '有无活动' in data.columns:
            data['是否活动时间'] = (1 - data['是否假期']) * data['有无活动']
        else:
            data['是否活动时间'] = 0

        # 处理缺失的原始列，设置为0或默认值
        required_original_cols = [
            '是否周末', '是否假期', '是否上课时间',
            '有无体育课', '有无活动', '温度', '风速', '湿度', '日照时间'
        ]

        for col in required_original_cols:
            if col not in data.columns:
                data[col] = 0

        # 历史统计特征 - 对于新数据，我们无法计算历史统计，设为0
        historical_features = [
            '历史平均人数', '历史最大人数', '历史标准差',
            '日平均人数', '人数_lag1', '人数_lag2', '人数_lag3', '人数_lag6'
        ]

        for feature in historical_features:
            data[feature] = 0

        return data


class TrafficFlowPredictorApp:
    """交通流量预测应用"""

    def __init__(self, root):
        self.root = root
        self.root.title("交通流量智能预测系统")
        self.root.geometry("1400x900")

        # 加载模型
        self.model = None
        self.feature_names = None
        self.load_model()

        # 创建界面
        self.create_widgets()

        # 初始化数据
        self.original_data = None
        self.processed_data = None
        self.current_sample_index = 0

        # 记录原始数据列
        self.original_columns = [
            '日期', '时间段', '人数', '是否高峰', '星期', '季节', '天气',
            '温度', '风速', '湿度', '日照时间', '空气状况',
            '是否周末', '是否假期', '是否上课时间', '有无体育课', '有无活动'
        ]

    def load_model(self):
        """加载训练好的模型"""
        try:
            model_data = joblib.load('xgboost_ensemble_traffic_model.pkl')
            self.model = {
                'classifier': model_data['classifier'],
                'normal_regressor': model_data['normal_regressor'],
                'peak_regressor': model_data['peak_regressor'],
                'feature_names': model_data['feature_names']
            }
            print(f"模型加载成功，特征数量: {len(self.model['feature_names'])}")
            print(f"模型特征: {self.model['feature_names'][:10]}...")  # 显示前10个特征
        except Exception as e:
            print(f"模型加载失败: {e}")
            messagebox.showerror("错误", f"模型加载失败: {e}")

    def create_widgets(self):
        """创建界面组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # 标题
        title_label = ttk.Label(main_frame, text="交通流量智能预测系统",
                                font=("微软雅黑", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))

        # 控制面板框架
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # 数据加载按钮
        ttk.Button(control_frame, text="加载原始数据",
                   command=self.load_data).grid(row=0, column=0, padx=5, pady=5)

        # 样本导航
        ttk.Label(control_frame, text="样本:").grid(row=0, column=1, padx=5)
        self.sample_spinbox = ttk.Spinbox(control_frame, from_=0, to=100, width=10,
                                          command=self.update_sample)
        self.sample_spinbox.grid(row=0, column=2, padx=5)
        self.sample_spinbox.set("0")

        ttk.Button(control_frame, text="上一页",
                   command=self.prev_sample).grid(row=0, column=3, padx=5)
        ttk.Button(control_frame, text="下一页",
                   command=self.next_sample).grid(row=0, column=4, padx=5)

        # 预测按钮
        ttk.Button(control_frame, text="预测当前样本",
                   command=self.predict_current).grid(row=0, column=5, padx=5)

        # 批量预测按钮
        ttk.Button(control_frame, text="批量预测",
                   command=self.batch_predict).grid(row=0, column=6, padx=5)

        # 创建选项卡
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 选项卡1: 原始数据预览
        self.data_frame = ttk.Frame(notebook)
        notebook.add(self.data_frame, text="原始数据")
        self.create_data_preview()

        # 选项卡2: 特征输入
        self.input_frame = ttk.Frame(notebook)
        notebook.add(self.input_frame, text="特征输入")
        self.create_feature_input()

        # 选项卡3: 预测结果
        self.result_frame = ttk.Frame(notebook)
        notebook.add(self.result_frame, text="预测结果")
        self.create_result_display()

        # 选项卡4: 可视化
        self.viz_frame = ttk.Frame(notebook)
        notebook.add(self.viz_frame, text="可视化分析")
        self.create_visualization()

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪 - 请加载原始数据")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

    def create_data_preview(self):
        """创建原始数据预览区域"""
        # 创建Treeview显示数据
        columns = ["序号", "日期", "时间段", "人数", "是否高峰", "星期", "季节", "天气",
                   "温度", "风速", "湿度", "日照时间", "空气状况", "是否周末",
                   "是否假期", "是否上课时间", "有无体育课", "有无活动"]

        tree_frame = ttk.Frame(self.data_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建滚动条
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        hsb.pack(side=tk.BOTTOM, fill=tk.X)

        # 创建Treeview
        self.data_tree = ttk.Treeview(tree_frame, columns=columns,
                                      yscrollcommand=vsb.set,
                                      xscrollcommand=hsb.set,
                                      show="headings")

        # 配置列
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=80, anchor=tk.CENTER)

        self.data_tree.pack(fill=tk.BOTH, expand=True)

        # 配置滚动条
        vsb.config(command=self.data_tree.yview)
        hsb.config(command=self.data_tree.xview)

        # 数据信息标签
        info_frame = ttk.Frame(self.data_frame)
        info_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.data_info_var = tk.StringVar()
        self.data_info_var.set("未加载数据")
        ttk.Label(info_frame, textvariable=self.data_info_var).pack(side=tk.LEFT)

    def create_feature_input(self):
        """创建原始特征输入区域"""
        # 创建滚动框架
        canvas = tk.Canvas(self.input_frame)
        scrollbar = ttk.Scrollbar(self.input_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 原始特征输入控件
        self.original_feature_vars = {}

        # 原始特征分组
        feature_groups = {
            "时间特征": ['日期', '时间段', '星期', '季节', '是否周末', '是否假期'],
            "活动特征": ['是否上课时间', '有无体育课', '有无活动'],
            "天气特征": ['温度', '风速', '湿度', '日照时间', '天气', '空气状况'],
            "目标值": ['人数', '是否高峰']
        }

        row = 0
        for group_name, features in feature_groups.items():
            # 分组标签
            ttk.Label(scrollable_frame, text=group_name,
                      font=("微软雅黑", 11, "bold")).grid(row=row, column=0,
                                                          columnspan=3, pady=(10, 5), sticky=tk.W)
            row += 1

            # 分组中的特征
            for i, feature in enumerate(features):
                col = i % 3
                if col == 0:
                    current_row = row
                    row += 1

                # 创建标签和输入框
                ttk.Label(scrollable_frame, text=f"{feature}:").grid(
                    row=current_row, column=col * 2, padx=5, pady=2, sticky=tk.W)

                if feature in ['日期', '时间段', '星期', '季节', '天气', '空气状况']:
                    # 文本输入
                    var = tk.StringVar(value="")
                    entry = ttk.Entry(scrollable_frame, textvariable=var, width=15)
                elif feature in ['是否高峰', '是否周末', '是否假期', '是否上课时间', '有无体育课', '有无活动']:
                    # 0/1输入
                    var = tk.IntVar(value=0)
                    entry = ttk.Entry(scrollable_frame, textvariable=var, width=15)
                else:
                    # 数值输入
                    var = tk.DoubleVar(value=0.0)
                    entry = ttk.Entry(scrollable_frame, textvariable=var, width=15)

                entry.grid(row=current_row, column=col * 2 + 1, padx=5, pady=2)
                self.original_feature_vars[feature] = var

        # 预测按钮
        ttk.Button(scrollable_frame, text="从输入预测",
                   command=self.predict_from_input).grid(
            row=row + 1, column=0, columnspan=6, pady=20)

        # 包装画布和滚动条
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 默认值按钮
        ttk.Button(self.input_frame, text="加载样本默认值",
                   command=self.load_default_values).pack(side=tk.BOTTOM, pady=10)

    def create_result_display(self):
        """创建预测结果显示区域"""
        # 结果框架
        result_frame = ttk.Frame(self.result_frame)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 预测结果
        result_text_frame = ttk.LabelFrame(result_frame, text="预测结果", padding="10")
        result_text_frame.pack(fill=tk.BOTH, expand=True)

        # 创建文本部件显示结果
        self.result_text = tk.Text(result_text_frame, height=15, width=60)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(result_text_frame, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)

        # 初始化文本
        self.result_text.insert(tk.END, "等待预测...\n")
        self.result_text.config(state=tk.DISABLED)

        # 统计信息框架
        stats_frame = ttk.Frame(result_frame)
        stats_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(stats_frame, text="统计信息:",
                  font=("微软雅黑", 11, "bold")).pack(anchor=tk.W)

        self.stats_text = tk.Text(stats_frame, height=5, width=60)
        self.stats_text.pack(fill=tk.X)
        self.stats_text.insert(tk.END, "暂无统计信息\n")
        self.stats_text.config(state=tk.DISABLED)

    def create_visualization(self):
        """创建可视化区域"""
        # 创建matplotlib图形
        self.viz_fig = Figure(figsize=(10, 6), dpi=100)

        # 创建画布
        self.viz_canvas = FigureCanvasTkAgg(self.viz_fig, master=self.viz_frame)
        self.viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 初始化空图
        ax = self.viz_fig.add_subplot(111)
        ax.text(0.5, 0.5, "等待数据可视化...",
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        self.viz_canvas.draw()

        # 可视化控制按钮
        control_frame = ttk.Frame(self.viz_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Button(control_frame, text="显示预测对比",
                   command=lambda: self.update_visualization('comparison')).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="显示残差分析",
                   command=lambda: self.update_visualization('residuals')).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="显示误差分布",
                   command=lambda: self.update_visualization('error_dist')).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="显示概率分布",
                   command=lambda: self.update_visualization('prob_dist')).pack(side=tk.LEFT, padx=5)

    def load_data(self):
        """加载原始数据文件"""
        file_path = filedialog.askopenfilename(
            title="选择原始数据文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )

        if file_path:
            try:
                # 读取原始数据
                self.original_data = pd.read_csv(file_path, encoding='utf-8-sig')
                self.status_var.set(f"原始数据加载成功: {len(self.original_data)} 条记录")

                # 预处理数据（生成特征）
                self.processed_data = TrafficDataPreprocessor.preprocess_input_data(self.original_data)

                # 确保所有模型需要的特征都存在
                if self.model and self.model['feature_names']:
                    for feature in self.model['feature_names']:
                        if feature not in self.processed_data.columns:
                            self.processed_data[feature] = 0
                            print(f"警告: 添加缺失特征 {feature}")

                # 更新数据预览
                self.update_data_preview()

                # 更新样本导航
                self.sample_spinbox.config(to=len(self.original_data) - 1)

                messagebox.showinfo("成功",
                                    f"成功加载 {len(self.original_data)} 条原始数据\n已生成 {len(self.processed_data.columns)} 个特征")

            except Exception as e:
                messagebox.showerror("错误", f"数据加载失败: {e}")

    def update_data_preview(self):
        """更新原始数据预览"""
        if self.original_data is None:
            return

        # 清空现有数据
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)

        # 添加新数据
        for i, row in self.original_data.iterrows():
            values = [i]
            for col in self.original_columns:
                if col in self.original_data.columns:
                    values.append(row[col])
                else:
                    values.append('')
            self.data_tree.insert("", tk.END, values=values)

        # 更新数据信息
        if '人数' in self.original_data.columns and '是否高峰' in self.original_data.columns:
            peak_count = self.original_data['是否高峰'].sum()
            self.data_info_var.set(
                f"总记录数: {len(self.original_data)} | 高峰样本: {peak_count} | 平均人数: {self.original_data['人数'].mean():.1f}")
        else:
            self.data_info_var.set(f"总记录数: {len(self.original_data)}")

    def load_default_values(self):
        """加载当前样本的默认特征值"""
        if self.original_data is not None and self.current_sample_index < len(self.original_data):
            sample = self.original_data.iloc[self.current_sample_index]

            # 设置原始特征值
            for feature, var in self.original_feature_vars.items():
                if feature in sample:
                    value = sample[feature]
                    if isinstance(var, tk.StringVar):
                        var.set(str(value))
                    elif isinstance(var, tk.IntVar):
                        try:
                            var.set(int(value))
                        except:
                            var.set(0)
                    elif isinstance(var, tk.DoubleVar):
                        try:
                            var.set(float(value))
                        except:
                            var.set(0.0)

            self.status_var.set(f"已加载样本 {self.current_sample_index} 的原始特征值")

    def predict_current(self):
        """预测当前选中的样本"""
        if self.original_data is None:
            messagebox.showwarning("警告", "请先加载原始数据")
            return

        if self.current_sample_index >= len(self.original_data):
            messagebox.showwarning("警告", "样本索引超出范围")
            return

        # 获取当前样本
        sample = self.original_data.iloc[self.current_sample_index].copy()

        # 预处理单个样本
        sample_df = pd.DataFrame([sample])
        processed_sample = TrafficDataPreprocessor.preprocess_input_data(sample_df)

        # 确保所有模型需要的特征都存在
        if self.model and self.model['feature_names']:
            for feature in self.model['feature_names']:
                if feature not in processed_sample.columns:
                    processed_sample[feature] = 0

        # 获取真实值
        true_value = sample.get('人数', 0) if '人数' in sample else None
        true_is_peak = sample.get('是否高峰', 0) if '是否高峰' in sample else None

        # 进行预测
        self.predict_and_display(processed_sample, true_value, true_is_peak)

    def predict_from_input(self):
        """从输入特征进行预测"""
        # 检查模型是否加载
        if self.model is None:
            messagebox.showerror("错误", "模型未加载，无法进行预测")
            return

        # 从输入获取原始特征值
        sample_data = {}
        for feature, var in self.original_feature_vars.items():
            if isinstance(var, tk.StringVar):
                sample_data[feature] = var.get()
            elif isinstance(var, tk.IntVar):
                sample_data[feature] = var.get()
            elif isinstance(var, tk.DoubleVar):
                sample_data[feature] = var.get()

        # 创建DataFrame
        sample_df = pd.DataFrame([sample_data])

        # 预处理
        processed_sample = TrafficDataPreprocessor.preprocess_input_data(sample_df)

        # 确保所有模型需要的特征都存在
        if self.model['feature_names'] is not None:
            for feature in self.model['feature_names']:
                if feature not in processed_sample.columns:
                    processed_sample[feature] = 0

        # 进行预测
        self.predict_and_display(processed_sample)

    def batch_predict(self):
        """批量预测"""
        if self.original_data is None:
            messagebox.showwarning("警告", "请先加载原始数据")
            return

        if self.processed_data is None:
            messagebox.showwarning("警告", "数据未预处理")
            return

        # 确保所有模型需要的特征都存在
        if self.model is None or self.model['feature_names'] is None:
            messagebox.showerror("错误", "模型特征未定义")
            return

        feature_names = self.model['feature_names']

        # 检查缺失特征
        missing_features = [f for f in feature_names if f not in self.processed_data.columns]
        if missing_features:
            for feature in missing_features:
                self.processed_data[feature] = 0

        # 重新排序特征
        X = self.processed_data[feature_names]

        # 获取真实值
        y_reg_true = self.original_data['人数'].values if '人数' in self.original_data.columns else None
        y_class_true = self.original_data['是否高峰'].values if '是否高峰' in self.original_data.columns else None

        # 批量预测
        predictions = []
        is_peak_list = []
        class_probs_list = []

        for i in range(len(X)):
            x_i = X.iloc[i:i + 1]

            # 分类器预测
            if self.model['classifier'] is not None:
                class_prob = self.model['classifier'].predict_proba(x_i)[0, 1]
                is_peak = 1 if class_prob > 0.4 else 0
            else:
                class_prob = 0.5
                is_peak = 0

            # 回归器预测
            if is_peak == 1 and self.model['peak_regressor'] is not None:
                pred = self.model['peak_regressor'].predict(x_i)[0]
            elif is_peak == 0 and self.model['normal_regressor'] is not None:
                pred = self.model['normal_regressor'].predict(x_i)[0]
            else:
                # 使用平均预测
                preds = []
                if self.model['normal_regressor'] is not None:
                    preds.append(self.model['normal_regressor'].predict(x_i)[0])
                if self.model['peak_regressor'] is not None:
                    preds.append(self.model['peak_regressor'].predict(x_i)[0])
                pred = np.mean(preds) if preds else 0

            predictions.append(max(pred, 0))
            is_peak_list.append(is_peak)
            class_probs_list.append(class_prob)

        # 显示统计结果
        self.show_batch_results(predictions, is_peak_list, class_probs_list,
                                y_reg_true, y_class_true)

    def predict_and_display(self, X, true_value=None, true_is_peak=None):
        """执行预测并显示结果"""
        if self.model is None:
            messagebox.showerror("错误", "模型未加载")
            return

        try:
            # 确保特征顺序正确
            if self.model['feature_names'] is not None:
                feature_names = self.model['feature_names']

                # 检查缺失特征
                missing_features = [f for f in feature_names if f not in X.columns]
                if missing_features:
                    for feature in missing_features:
                        X[feature] = 0

                # 重新排序特征
                X = X[feature_names]

            # 分类器预测
            if self.model['classifier'] is not None:
                class_probs = self.model['classifier'].predict_proba(X)[:, 1]
                is_peak = (class_probs > 0.4).astype(int)
            else:
                class_probs = np.array([0.5])
                is_peak = np.array([0])

            # 回归器预测
            predictions = np.zeros(len(X))

            for i in range(len(X)):
                x_i = X.iloc[i:i + 1]

                if is_peak[i] == 1 and self.model['peak_regressor'] is not None:
                    predictions[i] = self.model['peak_regressor'].predict(x_i)[0]
                elif is_peak[i] == 0 and self.model['normal_regressor'] is not None:
                    predictions[i] = self.model['normal_regressor'].predict(x_i)[0]
                else:
                    # 使用平均预测
                    preds = []
                    if self.model['normal_regressor'] is not None:
                        preds.append(self.model['normal_regressor'].predict(x_i)[0])
                    if self.model['peak_regressor'] is not None:
                        preds.append(self.model['peak_regressor'].predict(x_i)[0])
                    predictions[i] = np.mean(preds) if preds else 0

            predictions = np.maximum(predictions, 0)

            # 显示结果
            self.display_results(predictions[0], is_peak[0], class_probs[0],
                                 true_value, true_is_peak)

            # 更新可视化
            if true_value is not None and true_is_peak is not None:
                # 如果有历史数据，更新可视化
                if hasattr(self, 'batch_predictions') and self.batch_predictions is not None:
                    self.update_visualization('comparison')

        except Exception as e:
            messagebox.showerror("预测错误", f"预测失败: {e}")

    def display_results(self, pred_value, is_peak, class_prob,
                        true_value=None, true_is_peak=None):
        """显示单个预测结果"""
        # 启用文本编辑
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)

        # 构建结果文本
        result_text = "=" * 50 + "\n"
        result_text += "预测结果\n"
        result_text += "=" * 50 + "\n\n"

        result_text += f"预测人流量: {pred_value:.2f}\n"
        result_text += f"高峰概率: {class_prob:.4f}\n"
        result_text += f"是否高峰: {'是' if is_peak == 1 else '否'}\n"

        if true_value is not None:
            result_text += f"\n真实人流量: {true_value:.2f}\n"
            result_text += f"预测误差: {abs(pred_value - true_value):.2f}\n"
            if true_value > 0:
                error_percent = abs(pred_value - true_value) / true_value * 100
                result_text += f"相对误差: {error_percent:.2f}%\n"

        if true_is_peak is not None:
            result_text += f"\n真实高峰状态: {'是' if true_is_peak == 1 else '否'}\n"
            classification_correct = (is_peak == true_is_peak)
            result_text += f"分类结果: {'正确' if classification_correct else '错误'}\n"

        result_text += "\n" + "=" * 50

        # 插入文本
        self.result_text.insert(tk.END, result_text)
        self.result_text.config(state=tk.DISABLED)

        # 更新状态
        self.status_var.set(f"预测完成: 流量={pred_value:.2f}, 高峰概率={class_prob:.2%}")

    def show_batch_results(self, predictions, is_peak_list, class_probs_list,
                           y_reg_true=None, y_class_true=None):
        """显示批量预测结果"""
        # 保存批量预测结果用于可视化
        self.batch_predictions = predictions
        self.batch_is_peak = is_peak_list
        self.batch_class_probs = class_probs_list
        self.batch_y_reg_true = y_reg_true
        self.batch_y_class_true = y_class_true

        # 启用文本编辑
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)

        # 构建结果文本
        result_text = "=" * 50 + "\n"
        result_text += "批量预测结果\n"
        result_text += "=" * 50 + "\n\n"

        result_text += f"总样本数: {len(predictions)}\n"
        result_text += f"高峰样本数: {sum(is_peak_list)} ({sum(is_peak_list) / len(is_peak_list) * 100:.1f}%)\n"
        result_text += f"平均预测流量: {np.mean(predictions):.2f}\n"
        result_text += f"最大预测流量: {np.max(predictions):.2f}\n"
        result_text += f"最小预测流量: {np.min(predictions):.2f}\n"

        if y_reg_true is not None:
            # 计算回归指标
            mae = np.mean(np.abs(np.array(predictions) - y_reg_true))
            mse = np.mean((np.array(predictions) - y_reg_true) ** 2)
            rmse = np.sqrt(mse)

            result_text += "\n回归性能:\n"
            result_text += f"  平均绝对误差 (MAE): {mae:.2f}\n"
            result_text += f"  均方根误差 (RMSE): {rmse:.2f}\n"

            if len(predictions) > 1:
                # 计算R²
                ss_res = np.sum((np.array(predictions) - y_reg_true) ** 2)
                ss_tot = np.sum((y_reg_true - np.mean(y_reg_true)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                result_text += f"  决定系数 (R²): {r2:.4f}\n"

        if y_class_true is not None:
            # 计算分类指标
            accuracy = np.mean(np.array(is_peak_list) == y_class_true)
            precision = np.sum((np.array(is_peak_list) == 1) & (y_class_true == 1)) / max(np.sum(is_peak_list), 1)
            recall = np.sum((np.array(is_peak_list) == 1) & (y_class_true == 1)) / max(np.sum(y_class_true == 1), 1)

            result_text += "\n分类性能:\n"
            result_text += f"  准确率: {accuracy:.4f}\n"
            result_text += f"  精确率: {precision:.4f}\n"
            result_text += f"  召回率: {recall:.4f}\n"

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                result_text += f"  F1分数: {f1:.4f}\n"

        result_text += "\n" + "=" * 50

        # 插入文本
        self.result_text.insert(tk.END, result_text)
        self.result_text.config(state=tk.DISABLED)

        # 更新统计信息
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)

        stats_text = "批量预测统计:\n"
        stats_text += f"- 预测范围: {np.min(predictions):.2f} ~ {np.max(predictions):.2f}\n"
        stats_text += f"- 标准差: {np.std(predictions):.2f}\n"
        stats_text += f"- 中位数: {np.median(predictions):.2f}\n"

        if y_reg_true is not None:
            errors = np.abs(np.array(predictions) - y_reg_true)
            stats_text += f"- 平均误差: {np.mean(errors):.2f}\n"
            stats_text += f"- 最大误差: {np.max(errors):.2f}\n"

        self.stats_text.insert(tk.END, stats_text)
        self.stats_text.config(state=tk.DISABLED)

        # 更新状态
        self.status_var.set(f"批量预测完成: {len(predictions)} 个样本")

        # 更新可视化
        self.update_visualization('comparison')

    def update_visualization(self, viz_type):
        """更新可视化图表"""
        # 清除现有图形
        self.viz_fig.clear()

        if viz_type == 'comparison' and hasattr(self, 'batch_predictions'):
            # 预测对比图
            ax = self.viz_fig.add_subplot(111)

            predictions = self.batch_predictions
            y_reg_true = self.batch_y_reg_true

            if predictions is not None and y_reg_true is not None:
                n_show = min(50, len(predictions))
                indices = range(n_show)

                ax.plot(indices, y_reg_true[:n_show], 'b-', label='真实值', linewidth=2)
                ax.plot(indices, predictions[:n_show], 'r--', label='预测值', linewidth=2)
                ax.set_xlabel('样本索引')
                ax.set_ylabel('人流量')
                ax.set_title('预测值与真实值对比')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "无对比数据可用",
                        ha='center', va='center', fontsize=14)
                ax.set_axis_off()

        elif viz_type == 'residuals' and hasattr(self, 'batch_predictions'):
            # 残差分析图
            ax = self.viz_fig.add_subplot(111)

            predictions = self.batch_predictions
            y_reg_true = self.batch_y_reg_true

            if predictions is not None and y_reg_true is not None:
                residuals = predictions - y_reg_true
                ax.scatter(predictions, residuals, alpha=0.5, s=20)
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
                ax.set_xlabel('预测值')
                ax.set_ylabel('残差')
                ax.set_title('残差分析')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "无残差数据可用",
                        ha='center', va='center', fontsize=14)
                ax.set_axis_off()

        elif viz_type == 'error_dist' and hasattr(self, 'batch_predictions'):
            # 误差分布图
            ax = self.viz_fig.add_subplot(111)

            predictions = self.batch_predictions
            y_reg_true = self.batch_y_reg_true

            if predictions is not None and y_reg_true is not None:
                errors = np.abs(predictions - y_reg_true)
                ax.hist(errors, bins=20, alpha=0.7, edgecolor='black')
                ax.axvline(x=np.mean(errors), color='r', linestyle='--',
                           label=f'平均误差: {np.mean(errors):.2f}')
                ax.set_xlabel('绝对误差')
                ax.set_ylabel('频数')
                ax.set_title('预测误差分布')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "无误差数据可用",
                        ha='center', va='center', fontsize=14)
                ax.set_axis_off()

        elif viz_type == 'prob_dist' and hasattr(self, 'batch_class_probs'):
            # 概率分布图
            ax = self.viz_fig.add_subplot(111)

            class_probs = self.batch_class_probs
            y_class_true = self.batch_y_class_true

            if class_probs is not None and y_class_true is not None:
                normal_probs = class_probs[y_class_true == 0]
                peak_probs = class_probs[y_class_true == 1]

                if len(normal_probs) > 0:
                    ax.hist(normal_probs, bins=20, alpha=0.5, label='正常样本',
                            color='blue', density=True)
                if len(peak_probs) > 0:
                    ax.hist(peak_probs, bins=20, alpha=0.5, label='高峰样本',
                            color='red', density=True)
                ax.axvline(x=0.4, color='k', linestyle='--', alpha=0.7,
                           label='阈值(0.4)')
                ax.set_xlabel('高峰概率')
                ax.set_ylabel('密度')
                ax.set_title('分类概率分布')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "无概率数据可用",
                        ha='center', va='center', fontsize=14)
                ax.set_axis_off()

        else:
            ax = self.viz_fig.add_subplot(111)
            ax.text(0.5, 0.5, "请先进行批量预测以获得数据",
                    ha='center', va='center', fontsize=14)
            ax.set_axis_off()

        # 调整布局并重绘
        self.viz_fig.tight_layout()
        self.viz_canvas.draw()

    def update_sample(self):
        """更新当前样本索引"""
        try:
            self.current_sample_index = int(self.sample_spinbox.get())
            self.status_var.set(f"当前样本: {self.current_sample_index}")
        except:
            pass

    def prev_sample(self):
        """选择上一个样本"""
        if self.original_data is not None:
            new_index = max(0, self.current_sample_index - 1)
            self.current_sample_index = new_index
            self.sample_spinbox.set(str(new_index))
            self.status_var.set(f"当前样本: {self.current_sample_index}")

    def next_sample(self):
        """选择下一个样本"""
        if self.original_data is not None:
            new_index = min(len(self.original_data) - 1, self.current_sample_index + 1)
            self.current_sample_index = new_index
            self.sample_spinbox.set(str(new_index))
            self.status_var.set(f"当前样本: {self.current_sample_index}")


def main():
    """主函数"""
    root = tk.Tk()
    app = TrafficFlowPredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
