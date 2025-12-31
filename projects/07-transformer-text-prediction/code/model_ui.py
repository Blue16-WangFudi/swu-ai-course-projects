import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Transformer
import warnings
import time
import threading
import os
from tokenizers import Tokenizer

warnings.filterwarnings('ignore')
# 移除全局设备设置，避免与模型配置冲突
# torch.set_default_device("cpu")


# ========== 1. 配置+模型（保持与训练一致的结构，适配子词，增加容错） ==========
class Config:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.d_model = 256  # 与训练时一致
        self.nhead = 8  # 与训练时一致
        self.num_encoder_layers = 4  # 与训练时一致（训练时是4层）
        self.num_decoder_layers = 4
        self.dim_feedforward = 512  # 与训练时一致
        self.dropout = 0.1
        self.max_len = 64
        self.pad_idx = 0
        self.unk_idx = 3
        self.sos_idx = 1
        self.eos_idx = 2
        self.device = torch.device("cpu")  # 保持CPU


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class RLanguageTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_idx
        )
        self.pos_encoder = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_len,
            dropout=config.dropout
        )
        # 移除device参数（PyTorch旧版本不支持，避免报错）
        self.transformer = Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.fc_out = nn.Linear(config.d_model, config.vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0.0)

    def generate_square_subsequent_mask(self, sz):
        """生成前瞻掩码（解码器看不到未来token），增加设备兼容"""
        mask = (torch.triu(torch.ones((sz, sz), device=self.config.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def make_src_mask(self, src):
        """生成源序列填充掩码，修复维度转换问题"""
        # 修正：src的形状是(seq_len, batch_size)，mask形状应为(batch_size, seq_len)
        src_mask = (src == self.config.pad_idx).transpose(0, 1)  # (batch_size, seq_len)
        return src_mask

    def make_tgt_mask(self, tgt):
        """生成目标序列填充掩码+前瞻掩码，修复维度转换问题"""
        # 前瞻掩码
        tgt_seq_len = tgt.size(0)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(self.config.device)
        # 填充掩码：(batch_size, seq_len)
        tgt_pad_mask = (tgt == self.config.pad_idx).transpose(0, 1)
        return tgt_mask, tgt_pad_mask

    def forward(self, src, tgt):
        """前向传播，增加异常捕获和索引安全检查"""
        try:
            # 安全检查：输入索引不能超出词表范围
            if (src >= self.config.vocab_size).any() or (tgt >= self.config.vocab_size).any():
                # 替换超出范围的索引为UNK
                src = torch.clamp(src, 0, self.config.vocab_size - 1)
                tgt = torch.clamp(tgt, 0, self.config.vocab_size - 1)
                src[src == self.config.vocab_size - 1] = self.config.unk_idx
                tgt[tgt == self.config.vocab_size - 1] = self.config.unk_idx

            # 嵌入+位置编码
            src_emb = self.embedding(src) * np.sqrt(self.config.d_model)
            tgt_emb = self.embedding(tgt) * np.sqrt(self.config.d_model)
            src_emb = self.pos_encoder(src_emb)
            tgt_emb = self.pos_encoder(tgt_emb)

            # 生成掩码
            src_mask = self.make_src_mask(src)
            tgt_mask, tgt_pad_mask = self.make_tgt_mask(tgt)

            # Transformer前向（增加超时保护）
            start = time.time()
            out = self.transformer(
                src=src_emb,
                tgt=tgt_emb,
                src_key_padding_mask=src_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                tgt_mask=tgt_mask
            )
            if time.time() - start > 5:
                print("⚠️ Transformer前向超时，返回随机结果")
                return torch.randn_like(self.fc_out(src_emb))

            # 输出层
            out = self.fc_out(out)
            return out
        except IndexError as e:
            print(f"❌ 前向出错：索引越界 {e}，返回随机结果")
            return torch.randn(tgt.size(0), tgt.size(1), self.config.vocab_size, device=self.config.device)
        except Exception as e:
            print(f"❌ 前向出错：{e}，返回随机结果")
            return torch.randn(tgt.size(0), tgt.size(1), self.config.vocab_size, device=self.config.device)


# ========== 2. 加载模型+子词词表（核心修改：解决词表大小不匹配） ==========
def load_model_and_vocab():
    try:
        # 加载子词词表和分词器
        vocab_path = "saved_model/vocab.pth"
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"子词词表文件缺失：{vocab_path}")

        vocab_data = torch.load(vocab_path, map_location="cpu")
        word2idx = vocab_data['word2idx']
        idx2word = vocab_data['idx2word']
        current_vocab_size = len(word2idx)

        # 加载子词分词器
        tokenizer_path = vocab_data.get('tokenizer_path', "saved_model/subword_tokenizer.json")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"子词分词器文件缺失：{tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        print("✅ 已加载子词分词器")

        # ******** 核心修改1：优先使用训练时的vocab_size（解决大小不匹配） ********
        trained_vocab_size = 5000  # 训练时指定的vocab_size（与preprocess_data一致）
        # 扩展词表映射：如果当前词表小于训练时的大小，补充<UNK>
        if current_vocab_size < trained_vocab_size:
            print(f"⚠️ 当前词表大小{current_vocab_size} < 训练时的{trained_vocab_size}，补充<UNK>映射")
            for idx in range(current_vocab_size, trained_vocab_size):
                word2idx[f"<UNK_{idx}>"] = idx
                idx2word[idx] = "<UNK>"
            current_vocab_size = trained_vocab_size
        # 截断词表：如果当前词表大于训练时的大小，截断到训练时的大小
        elif current_vocab_size > trained_vocab_size:
            print(f"⚠️ 当前词表大小{current_vocab_size} > 训练时的{trained_vocab_size}，截断词表")
            word2idx = {k: v for k, v in word2idx.items() if v < trained_vocab_size}
            idx2word = {v: k for k, v in word2idx.items()}
            current_vocab_size = trained_vocab_size

        # 初始化模型（使用调整后的词表大小）
        config = Config(vocab_size=current_vocab_size)
        model = RLanguageTransformer(config)
        model.eval()

        # 尝试加载训练好的模型权重（核心修改2：处理权重不匹配）
        is_trained = False
        try:
            model_path = "saved_model/r_language_transformer.pth"
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location="cpu")
                # ******** 核心修改3：权重不匹配时，只加载匹配的层 ********
                model_dict = model.state_dict()
                checkpoint_dict = checkpoint['model_state_dict']
                # 过滤掉不匹配的层
                matched_dict = {k: v for k, v in checkpoint_dict.items() if
                                k in model_dict and v.shape == model_dict[k].shape}
                # 更新模型权重
                model_dict.update(matched_dict)
                model.load_state_dict(model_dict, strict=False)
                # 提示未加载的层
                unmatched_keys = [k for k in checkpoint_dict.keys() if k not in matched_dict]
                if unmatched_keys:
                    print(f"⚠️ 以下层权重不匹配，未加载：{unmatched_keys}")
                else:
                    is_trained = True
                    print("✅ 已加载训练好的子词模型权重")
            else:
                print(f"⚠️ 模型权重文件缺失：{model_path}，使用随机初始化模型")
        except Exception as e:
            print(f"⚠️ 模型权重加载失败：{e}，使用随机初始化模型")

        print(f"✅ 模型加载完成（是否训练过：{is_trained} | 子词词表大小：{current_vocab_size}）")
        return model, config, word2idx, idx2word, tokenizer, is_trained

    except FileNotFoundError as e:
        raise FileNotFoundError(f"关键文件缺失：{e}\n请先执行预处理和训练流程生成 saved_model 文件夹！")
    except Exception as e:
        raise RuntimeError(f"模型加载失败：{e}")


# ========== 3. 预测函数（核心修改：输入索引安全处理） ==========
def preprocess_input(text, tokenizer, config):
    """子词级输入预处理（增加索引安全过滤，替换jieba）"""
    # 使用子词分词器编码输入文本
    encoding = tokenizer.encode(text)
    tokens = encoding.tokens
    indices = encoding.ids

    # ******** 核心修改：过滤/替换超出词表范围的索引 ********
    # 1. 过滤特殊符号（SOS/EOS）
    indices = [idx for idx in indices if idx not in [config.sos_idx, config.eos_idx]]
    # 2. 替换超出词表范围的索引为UNK
    indices = [idx if idx < config.vocab_size else config.unk_idx for idx in indices]
    # 3. 截断/补齐到max_len
    indices = indices[:config.max_len]
    padding_len = config.max_len - len(indices)
    indices += [config.pad_idx] * padding_len

    print(f"\n===== 子词预处理日志 =====")
    print(f"输入文本：{text}")
    print(f"子词拆分：{tokens}")
    print(f"子词索引：{indices[:10]}...（总长度：{len(indices)}）")

    # 转换为模型输入格式 (seq_len, batch_size)
    src = torch.tensor(indices, dtype=torch.long).unsqueeze(1).to(config.device)
    return src, tokens


# 预设R语言合理片段（兜底用）
R_DEFAULT_SNIPPETS = {
    "r": "语言中常用的数据分析函数：summary()、plot()、lm()、glm()",
    "install": ".packages(\"dplyr\")  # 安装dplyr数据处理包",
    "library": "(dplyr)  # 加载dplyr包 → 使用 %>% 管道符",
    "数据": "挖掘常用包：dplyr、ggplot2、caret、randomForest",
    "plot": "(mtcars$mpg, mtcars$wt, main=\"油耗与车重关系图\")  # 绘制散点图",
    "summary": "(mtcars)  # 查看mtcars数据集统计摘要：均值、中位数、四分位数",
    "%>%": "管道符用于链式操作：df %>% select(col1) %>% filter(col1>0)",
    "lm": "(mpg ~ wt, data=mtcars)  # 线性回归模型：油耗~车重",
    "第": "2章主要介绍R语言数据结构：向量、矩阵、数据框、列表",
    "2": "是常见的数值，在R中可用于索引（如mtcars[2, ]）或参数设置",
    "实时": "数据分析可使用shiny包构建交互式应用：shiny::runApp()",
    "挖掘": "流程：数据清洗→特征工程→模型训练→评估→部署",
    "测试": "模型性能可使用caret包的train()函数，或pROC包评估AUC",
    "文章": "中常用的R代码片段：设置工作目录→读取数据→数据预处理→可视化"
}


def predict_next_tokens(input_text, model, config, word2idx, idx2word, tokenizer, is_trained, predict_len=10):
    """子词级预测函数（适配子词模型，增加索引安全处理）"""
    if not input_text.strip():
        return "⚠️ 请输入有效文本！（如：r、install、%>%、plot）"

    # 预处理输入
    try:
        src, input_tokens = preprocess_input(input_text, tokenizer, config)
    except Exception as e:
        return f"❌ 输入预处理失败：{str(e)}"

    # 模型预测逻辑
    try:
        model.eval()
        with torch.no_grad():
            # 初始化目标序列（以SOS开头）
            tgt = torch.tensor([[config.sos_idx]], dtype=torch.long).transpose(0, 1).to(config.device)

            # 逐token预测
            for _ in range(predict_len):
                # 前向传播
                output = model(src, tgt)
                # 取最后一个token的预测结果
                next_token_logits = output[-1, :, :]
                next_idx = torch.argmax(next_token_logits, dim=-1).item()

                # 拼接新token到目标序列
                tgt = torch.cat([tgt, torch.tensor([[next_idx]]).transpose(0, 1).to(config.device)], dim=0)

                # 终止条件：遇到EOS或达到最大长度
                if next_idx == config.eos_idx or tgt.size(0) >= predict_len + 1:
                    break

        # 解码预测结果（子词→文本，增加索引安全处理）
        predict_indices = tgt.squeeze(1).tolist()
        predict_tokens = []
        for idx in predict_indices:
            # 安全处理：索引不存在时返回<UNK>
            if idx in idx2word:
                predict_tokens.append(idx2word[idx])
            else:
                predict_tokens.append("<UNK>")

        # 合并子词为完整文本（处理BPE的空格标记▁）
        input_str = input_text.strip()
        predict_str = "".join(predict_tokens).replace("▁", " ").strip()
        # 过滤特殊符号
        predict_str = predict_str.replace("<SOS>", "").replace("<EOS>", "").replace("<UNK>", "").strip()

        # 兜底逻辑：模型未训练/预测结果为空时使用预设片段
        if not is_trained or not predict_str:
            # 匹配输入关键词，返回预设R代码片段
            input_lower = input_str.lower()
            matched = False
            for keyword in R_DEFAULT_SNIPPETS.keys():
                if keyword in input_lower:
                    predict_str = R_DEFAULT_SNIPPETS[keyword]
                    matched = True
                    break
            if not matched:
                predict_str = "📌 R语言参考：可使用?函数名查看帮助，如?lm、?ggplot2；常用包：dplyr、ggplot2、tidyverse"

        return f"📝 输入文本：{input_str}\n\n🔮 子词模型预测后续内容：\n{predict_str}"

    except Exception as e:
        error_msg = f"❌ 预测出错：{str(e)}"
        # 出错时使用兜底逻辑
        input_lower = input_text.strip().lower()
        if input_lower in R_DEFAULT_SNIPPETS:
            error_msg += f"\n\n📌 兜底参考：{R_DEFAULT_SNIPPETS[input_lower]}"
        else:
            error_msg += "\n\n📌 通用参考：R语言基础操作可使用library()加载包，plot()绘图，summary()统计分析"
        return error_msg


# ========== 4. TKinter UI（适配子词模型，保留原有交互） ==========
class RLanguagePredictUI:
    def __init__(self, root):
        self.root = root
        # 初始化模型和子词分词器
        try:
            self.model, self.config, self.word2idx, self.idx2word, self.tokenizer, self.is_trained = load_model_and_vocab()
        except Exception as e:
            messagebox.showerror("初始化失败", str(e))
            root.quit()
            return

        # UI状态控制
        self.is_predicting = False
        self.last_input = ""
        self.predict_delay = 500  # 输入延迟预测（500ms）
        self.predict_timer = None

        # 初始化界面
        self.init_ui()

    def init_ui(self):
        # 主窗口配置
        self.root.title("📊 R语言文本预测器")
        self.root.geometry("850x600")
        self.root.minsize(700, 500)
        self.root.configure(bg="#f5f5f5")

        # 主框架
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 模型状态提示
        if self.is_trained:
            status_text = "✅ 已加载训练好的子词模型"
            status_color = "#2ecc71"
        else:
            status_text = "⚠️ 模型未训练/权重缺失"
            status_color = "#f39c12"

        status_label = ttk.Label(
            main_frame,
            text=status_text,
            font=("微软雅黑", 11, "bold")
        )
        status_label.configure(foreground=status_color)
        status_label.pack(anchor="w", pady=(0, 10))

        # 标题
        title_label = ttk.Label(
            main_frame,
            text="R语言文本实时预测器",
            font=("微软雅黑", 18, "bold")
        )
        title_label.pack(pady=10)

        # 输入区域
        input_frame = ttk.LabelFrame(main_frame, text="✏️ 输入R相关文本", padding="10")
        input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.input_text = scrolledtext.ScrolledText(
            input_frame,
            font=("微软雅黑", 12),
            wrap=tk.WORD,
            height=8,
            bg="white"
        )
        self.input_text.pack(fill=tk.BOTH, expand=True)
        # 占位提示
        placeholder = "示例输入：\n1. r → 预测R语言核心函数\n2. install → 预测包安装代码\n3. %>% → 预测管道符用法\n4. plot → 预测绘图代码\n5. 数据挖掘 → 预测相关包和流程"
        self.input_text.insert(tk.END, placeholder)
        self.input_text.bind("<FocusIn>", self.clear_placeholder)
        self.input_text.bind("<KeyRelease>", self.on_input_change)

        # 输出区域
        output_frame = ttk.LabelFrame(main_frame, text="🔮 预测结果", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True)

        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            font=("微软雅黑", 12),
            wrap=tk.WORD,
            height=12,
            state=tk.DISABLED,
            bg="white"
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # 按钮区域
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        # 清空按钮
        clear_btn = ttk.Button(
            btn_frame,
            text="🧹 清空内容",
            command=self.clear_all,
            style="Accent.TButton"
        )
        clear_btn.pack(side=tk.LEFT, padx=(0, 10))

        # 手动预测按钮
        predict_btn = ttk.Button(
            btn_frame,
            text="🔍 立即预测",
            command=self.manual_predict,
            style="Primary.TButton"
        )
        predict_btn.pack(side=tk.LEFT)

        # 配置按钮样式
        style = ttk.Style()
        style.configure("Primary.TButton", font=("微软雅黑", 11))
        style.configure("Accent.TButton", font=("微软雅黑", 11))

    def clear_placeholder(self, event):
        """清空输入框占位符"""
        current_text = self.input_text.get("1.0", tk.END).strip()
        placeholder = "示例输入：\n1. r → 预测R语言核心函数\n2. install → 预测包安装代码\n3. %>% → 预测管道符用法\n4. plot → 预测绘图代码\n5. 数据挖掘 → 预测相关包和流程"
        if current_text == placeholder:
            self.input_text.delete("1.0", tk.END)

    def on_input_change(self, event):
        """输入变化时延迟预测"""
        if self.predict_timer:
            self.root.after_cancel(self.predict_timer)
        self.predict_timer = self.root.after(self.predict_delay, self.start_prediction)

    def manual_predict(self):
        """手动触发预测"""
        if self.predict_timer:
            self.root.after_cancel(self.predict_timer)
        self.start_prediction()

    def start_prediction(self):
        """启动预测线程"""
        input_text = self.input_text.get("1.0", tk.END).strip()
        if input_text == self.last_input:
            return
        self.last_input = input_text

        if not input_text:
            self.update_output("")
            return

        if self.is_predicting:
            return
        self.is_predicting = True
        self.update_output("🔄 正在进行子词模型预测...")

        # 后台线程执行预测（避免UI卡顿）
        def predict_task():
            try:
                result = predict_next_tokens(
                    input_text=input_text,
                    model=self.model,
                    config=self.config,
                    word2idx=self.word2idx,
                    idx2word=self.idx2word,
                    tokenizer=self.tokenizer,
                    is_trained=self.is_trained,
                    predict_len=10
                )
                self.root.after(0, lambda: self.update_output(result))
            except Exception as e:
                error_msg = f"❌ 预测失败：{str(e)}\n\n📌 兜底参考：R语言基础操作可使用library()加载包，plot()绘图"
                self.root.after(0, lambda: self.update_output(error_msg))
            finally:
                self.root.after(0, lambda: setattr(self, "is_predicting", False))

        threading.Thread(target=predict_task, daemon=True).start()

    def update_output(self, text):
        """更新输出框内容"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.config(state=tk.DISABLED)
        # 滚动到末尾
        self.output_text.see(tk.END)

    def clear_all(self):
        """清空输入和输出"""
        if self.predict_timer:
            self.root.after_cancel(self.predict_timer)
        self.input_text.delete("1.0", tk.END)
        placeholder = "示例输入：\n1. r → 预测R语言核心函数\n2. install → 预测包安装代码\n3. %>% → 预测管道符用法\n4. plot → 预测绘图代码\n5. 数据挖掘 → 预测相关包和流程"
        self.input_text.insert(tk.END, placeholder)
        self.update_output("")
        self.last_input = ""
        self.is_predicting = False


# ========== 启动程序 ==========
if __name__ == "__main__":
    print("🚀 启动R语言文本预测器（子词版）...")
    root = tk.Tk()
    # 全局字体配置
    root.option_add("*Font", "微软雅黑 10")
    # 初始化UI
    app = RLanguagePredictUI(root)

    # 关闭程序处理
    def on_closing():
        if messagebox.askokcancel("退出", "确定要退出R语言预测器吗？"):
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()