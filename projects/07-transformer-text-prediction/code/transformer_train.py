import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import Transformer
from torch.utils.data import DataLoader
from tqdm import tqdm  # 进度条可视化
import warnings

warnings.filterwarnings('ignore')


# ========== 1. 配置参数（增强：适配更大词表，解耦硬编码参数） ==========
class Config:
    def __init__(self, vocab_size):
        # 模型核心参数（根据词表大小提升模型容量）
        self.vocab_size = vocab_size  # 子词词表大小（从预处理结果传入）
        self.d_model = 512  # 提升特征维度（原256，匹配更大词表）
        self.nhead = 8  # 保持8头（512是8的倍数，无需修改）
        self.num_encoder_layers = 6  # 恢复层数（原4层，提升模型表达能力）
        self.num_decoder_layers = 6  # 恢复层数
        self.dim_feedforward = 2048  # 提升前馈网络维度（原512）
        self.dropout = 0.1  # dropout率保持不变（防止过拟合）
        self.max_len = 128  # 提升序列最大长度（原64，适配更长子词序列）
        self.pad_idx = 0  # PAD token索引（与子词预处理一致）
        self.unk_idx = 3  # UNK token索引
        self.sos_idx = 1  # SOS token索引
        self.eos_idx = 2  # EOS token索引

        # 训练参数（适配更大模型与词表）
        self.epochs = 50  # 增加训练轮数（原30，适配更复杂模型）
        self.batch_size = 16  # 提升批次大小（原8，需根据设备内存调整）
        self.lr = 1e-4  # 初始学习率保持不变
        self.weight_decay = 1e-5  # 权重衰减保持不变
        self.patience = 8  # 增加早停耐心值（原5，适配更长训练）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备
        self.save_path = "saved_model"  # 模型保存路径
        self.model_name = "r_language_transformer_large_vocab.pth"  # 重命名模型（区分大词表版本）

    def print_config(self):
        """新增：打印配置信息（方便调试）"""
        print("\n⚙️  模型配置（适配大子词词表）：")
        for k, v in sorted(self.__dict__.items()):
            print(f"  {k}: {v}")


# ========== 2. 位置编码（无修改，保留核心逻辑） ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 生成位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # 偶数位用正弦
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # 奇数位用余弦
        self.register_buffer('pe', pe)  # 不参与梯度更新

    def forward(self, x):
        """
        x: 输入序列 (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# ========== 3. 完整Transformer模型（无核心修改，仅适配配置变化） ==========
class RLanguageTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 词嵌入层（适配更大子词词表）
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_idx
        )

        # 位置编码层（适配更大max_len）
        self.pos_encoder = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_len,
            dropout=config.dropout
        )

        # Transformer核心（PyTorch内置优化版，适配新参数）
        self.transformer = Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",  # 前馈层激活函数
            batch_first=False,
            norm_first=True  # 前置层归一化
        )

        # 输出层,映射回更大的子词词表
        self.fc_out = nn.Linear(config.d_model, config.vocab_size)

        # 初始化权重（提升收敛速度）
        self._init_weights()

    def _init_weights(self):
        """初始化嵌入层和输出层权重"""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0.0)

    def generate_square_subsequent_mask(self, sz):
        """生成前瞻掩码（解码器看不到未来token）"""
        mask = (torch.triu(torch.ones((sz, sz), device=self.config.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def make_src_mask(self, src):
        """生成源序列填充掩码（适配子词PAD索引）"""
        src_mask = (src == self.config.pad_idx).transpose(0, 1)  # (batch_size, seq_len)
        return src_mask

    def make_tgt_mask(self, tgt):
        """生成目标序列填充掩码+前瞻掩码（适配子词PAD索引）"""
        # 前瞻掩码
        tgt_seq_len = tgt.size(0)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(self.config.device)

        # 填充掩码
        tgt_pad_mask = (tgt == self.config.pad_idx).transpose(0, 1)  # (batch_size, seq_len)
        return tgt_mask, tgt_pad_mask

    def forward(self, src, tgt):
        """
        前向传播（适配子词序列输入）
        src: 源序列 (seq_len, batch_size)
        tgt: 目标序列 (seq_len, batch_size)
        """
        # 1. 嵌入+位置编码
        src_emb = self.pos_encoder(self.embedding(src) * np.sqrt(self.config.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * np.sqrt(self.config.d_model))

        # 2. 生成掩码
        src_mask = self.make_src_mask(src)
        tgt_mask, tgt_pad_mask = self.make_tgt_mask(tgt)

        # 3. Transformer前向
        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            tgt_mask=tgt_mask
        )

        # 4. 输出层（映射回子词词表）
        out = self.fc_out(out)
        return out


# ========== 4. 训练工具函数（无核心修改，保留逻辑） ==========
def train_epoch(model, dataloader, criterion, optimizer, config, epoch):
    """单轮训练"""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.epochs}")

    for src_batch, tgt_batch in progress_bar:
        # 调整维度：(batch_size, seq_len) → (seq_len, batch_size)（适配Transformer输入格式）
        src = src_batch.transpose(0, 1).to(config.device)
        tgt = tgt_batch.transpose(0, 1).to(config.device)

        # 目标输入：去掉最后一个token（tgt_in），目标标签：去掉第一个token（tgt_out）
        tgt_in = tgt[:-1, :]
        tgt_out = tgt[1:, :]

        # 前向传播
        optimizer.zero_grad()
        output = model(src, tgt_in)

        # 计算损失（忽略子词PAD token）
        loss = criterion(
            output.reshape(-1, config.vocab_size),
            tgt_out.reshape(-1)
        )

        # 反向传播+优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪（防止梯度爆炸）
        optimizer.step()

        # 更新进度条
        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Avg Loss": f"{total_loss / (progress_bar.n + 1):.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate_epoch(model, dataloader, criterion, config):
    """单轮验证（无梯度，适配子词序列）"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for src_batch, tgt_batch in dataloader:
            src = src_batch.transpose(0, 1).to(config.device)
            tgt = tgt_batch.transpose(0, 1).to(config.device)

            tgt_in = tgt[:-1, :]
            tgt_out = tgt[1:, :]

            output = model(src, tgt_in)
            loss = criterion(
                output.reshape(-1, config.vocab_size),
                tgt_out.reshape(-1)
            )
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# ========== 5. 主训练函数（无核心修改，保留逻辑） ==========
def train_transformer(dataloader, vocab_size, config):
    """
    完整训练流程：初始化模型→训练→早停→保存（适配子词分词）
    """
    # 1. 初始化模型、损失函数、优化器
    model = RLanguageTransformer(config).to(config.device)

    # 损失函数（忽略子词PAD token）
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx)

    # 优化器（AdamW + 权重衰减）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # 修复：移除verbose参数（兼容旧版PyTorch）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
        # 移除verbose参数，避免版本兼容问题
    )

    # 2. 早停初始化
    best_val_loss = float('inf')
    patience_counter = 0

    # 3. 创建保存目录
    os.makedirs(config.save_path, exist_ok=True)

    # 4. 训练循环
    print(f"\n🚀 开始训练（设备：{config.device}）")
    print(f"模型参数总量：{sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(config.epochs):
        # 训练
        train_loss = train_epoch(model, dataloader, criterion, optimizer, config, epoch)

        # 验证（用训练集的10%做简单验证，可替换为独立验证集）
        val_loss = validate_epoch(model, dataloader, criterion, config)

        # 学习率调度
        scheduler.step(val_loss)

        # 打印日志
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 早停逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型（适配子词配置）
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config.__dict__,
                'vocab_size': vocab_size
            }, os.path.join(config.save_path, config.model_name))
            print(f"✅ 保存最佳模型（验证损失：{best_val_loss:.4f}）")
        else:
            patience_counter += 1
            print(f"⚠️  验证损失未下降，耐心值：{patience_counter}/{config.patience}")
            if patience_counter >= config.patience:
                print(f"\n🛑 早停触发，训练结束！最佳验证损失：{best_val_loss:.4f}")
                break

    # 5. 加载最佳模型并返回
    best_model_path = os.path.join(config.save_path, config.model_name)
    checkpoint = torch.load(best_model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n🎉 训练完成！最佳模型已保存至：{best_model_path}")

    return model, checkpoint


# ========== 6. 入口函数（核心修改：提升子词词表大小，解耦配置） ==========
def main():
    # ====================== 核心修改1：配置大词表参数（可根据需求调整） ======================
    SUBWORD_VOCAB_SIZE = 32000  # 子词词表大小（从5000提升至32000，可根据数据量调整为16000/24000/40000）
    BATCH_SIZE = 16  # 批次大小（与Config中的batch_size保持一致）
    TEXT_FILE_PATH = "data_all.txt"  # 文本数据路径

    # 1. 导入适配子词的预处理模块（确保修改后的data_processed.py在同一目录）
    from data_processed import preprocess_data

    # 2. 执行子词级预处理，获取dataloader和子词词表大小
    print("📚 第一步：执行子词级文本预处理（大词表版本）...")
    print(f"🔍 子词词表大小设置为：{SUBWORD_VOCAB_SIZE}")
    dataloader, vocab, vocab_size, _, _ = preprocess_data(
        book_path=TEXT_FILE_PATH,
        batch_size=BATCH_SIZE,
        vocab_size=SUBWORD_VOCAB_SIZE  # 传入大词表大小（核心修改）
        # 移除custom_dict_path参数：子词分词无需自定义词典
    )

    # 3. 初始化配置（适配大子词词表）
    config = Config(vocab_size=vocab_size)
    config.print_config()  # 调用新增的打印函数

    # 4. 训练模型（适配子词序列）
    print("\n🔥 第二步：开始训练Transformer模型（大子词词表版本）...")
    model, checkpoint = train_transformer(dataloader, vocab_size, config)

    # 5. 保存子词词表（适配子词分词器结构）
    vocab_path = os.path.join(config.save_path, "vocab_large.pth")  # 重命名词表文件
    torch.save({
        'word2idx': vocab.word2idx,          # 子词→索引映射
        'idx2word': vocab.idx2word,          # 索引→子词映射
        'vocab_size': vocab_size,            # 新增：保存词表大小
        'tokenizer_path': os.path.join(config.save_path, "subword_tokenizer_large.json")  # 重命名分词器文件
    }, vocab_path)
    print(f"\n📖 大子词词表已保存至：{vocab_path}")


# ====================== 重要提醒：同步修改data_processed.py ======================
# 需确保data_processed.py中的子词分词器（如SentencePiece/BPE）的vocab_size参数与这里的SUBWORD_VOCAB_SIZE一致！
# 例如：在SentencePieceTrainer.Train时，设置--vocab_size={SUBWORD_VOCAB_SIZE}

if __name__ == "__main__":
    main()