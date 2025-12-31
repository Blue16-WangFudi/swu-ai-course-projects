import os
import re
import torch
import chardet
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
# 新增：子词分词依赖
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.processors import TemplateProcessing


# ===== 1. 文本规范化层（优化：保留段落换行分隔）=====
class TextNormalizationLayer:
    """文本规范化层：编码统一（UTF-8）、全角转半角、大小写统一、符号规范化"""

    def __init__(self):
        # R语言关键字（统一小写）
        self.r_keywords = ['library', 'require', 'data', 'function', 'if', 'else', 'for', 'while',
                           'lm', 'glm', 'plot', 'summary', 'dplyr', 'ggplot2', 'tidyverse']

    def detect_encoding(self, file_path):
        """自动检测文件编码（解决乱码问题）"""
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding'] or 'utf-8'
        # 兼容常见编码
        if encoding.lower() in ['gb2312', 'gbk']:
            encoding = 'gb18030'
        return encoding

    def full2half(self, text):
        """全角字符转半角（解决中文文本中的全角空格/符号）"""
        result = []
        for char in text:
            code = ord(char)
            # 全角空格转半角
            if code == 12288:
                code = 32
            # 全角字符（除空格）转半角
            elif 65281 <= code <= 65374:
                code -= 65248
            result.append(chr(code))
        return ''.join(result)

    def normalize_symbols(self, text):
        """规范化符号：仅清理水平空格→统一标点→保留换行分隔"""
        # ========== 仅清理水平空格（不影响换行） ==========
        # 1. 中文 + 空格 + 数字（如“第 2 章”→“第2章”）
        text = re.sub(r'([\u4e00-\u9fa5])\s+(\d+)\s*([\u4e00-\u9fa5]?)', r'\1\2\3', text)
        # 2. 数字 + 空格 + 中文（如“2 章”→“2章”）
        text = re.sub(r'(\d+)\s+([\u4e00-\u9fa5])', r'\1\2', text)
        # 3. 中文 + 空格 + 英文（如“R 语言”→“R语言”）
        text = re.sub(r'([a-zA-Z0-9_])\s+([\u4e00-\u9fa5])', r'\1\2', text)
        # 4. 英文 + 空格 + 中文（如“语言 R”→“语言R”）
        text = re.sub(r'([\u4e00-\u9fa5])\s+([a-zA-Z0-9_])', r'\1\2', text)

        # 5. 文字 + 空格 + 标点（如“本章 .”→“本章.”）
        text = re.sub(r'([\u4e00-\u9fa5a-zA-Z0-9])\s+([\.\,\;\:\!\?\。\，\；\：\！\？])', r'\1\2', text)
        # 6. 标点 + 空格 + 文字（如“. 读者”→“.读者”）
        text = re.sub(r'([\.\,\;\:\!\?\。\，\；\：\！\？])\s+([\u4e00-\u9fa5a-zA-Z0-9])', r'\1\2', text)
        # 7. 清理连续多个标点（仅水平空格）
        text = re.sub(r'([\.\,\;\:\!\?]){2,}', r'\1', text)
        # 关键：只替换水平空格（[ \t]），不包含换行符（\n）
        text = re.sub(r'[ \t]+', ' ', text)

        # ========== 标点统一 + 保留换行 ==========
        # 统一R符号/中文标点转半角
        text = re.sub(r'＜－', '<-', text)
        text = re.sub(r'％＞％', '%>%', text)
        text = re.sub(r'。', '.', text)
        text = re.sub(r'，', ',', text)
        text = re.sub(r'；', ';', text)
        text = re.sub(r'：', ':', text)
        text = re.sub(r'（', '(', text)
        text = re.sub(r'）', ')', text)

        # 8. 删除中文间水平空格（不影响换行）
        text = re.sub(r'([\u4e00-\u9fa5])[ \t]+([\u4e00-\u9fa5])', r'\1\2', text)
        while re.search(r'([\u4e00-\u9fa5])[ \t]+([\u4e00-\u9fa5])', text):
            text = re.sub(r'([\u4e00-\u9fa5])[ \t]+([\u4e00-\u9fa5])', r'\1\2', text)

        # R关键字统一小写
        for keyword in self.r_keywords:
            text = re.sub(rf'\b{keyword.upper()}\b', keyword, text)
        return text

    def normalize(self, file_path):
        """执行完整规范化流程：保留段落换行分隔"""
        # 1. 检测文件编码
        encoding = self.detect_encoding(file_path)
        # 2. 读取文件（兼容不同编码）
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                raw_text = f.read()
        except (UnicodeDecodeError, LookupError):
            # 兜底：用utf-8忽略错误
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
        # 3. 全角转半角
        raw_text = self.full2half(raw_text)
        # 4. 符号规范化（仅清理水平空格，保留换行）
        raw_text = self.normalize_symbols(raw_text)
        # 5. 仅清理连续3个及以上的换行，保留\n\n段落分隔
        raw_text = re.sub(r'\n{3,}', '\n\n', raw_text)
        # 补充：移除每行首尾的水平空格（不影响换行）
        raw_text = re.sub(r'^[ \t]+|[ \t]+$', '', raw_text, flags=re.MULTILINE)
        return raw_text.strip()


# ===== 2. 数据清洗层（核心修改：保留段落换行分隔）=====
class DataCleaningLayer:
    """数据清洗层：移除缺失值、清理异常格式，保留段落换行分隔"""

    def __init__(self, min_paragraph_len=5):
        self.min_paragraph_len = min_paragraph_len  # 最小段落长度阈值

    def remove_missing_values(self, text):
        """移除缺失值：仅清理空段落，保留非空段落的换行分隔"""
        # 按\n\n拆分段落（保留原有段落结构）
        paragraphs = text.split('\n\n')
        cleaned_paragraphs = []
        for para in paragraphs:
            # 仅清理段落内的水平空格，不影响换行
            clean_para = re.sub(r'([\u4e00-\u9fa5a-zA-Z0-9])\s+([\.\,\;\:\!\?])', r'\1\2', para)
            clean_para = re.sub(r'([\.\,\;\:\!\?])\s+([\u4e00-\u9fa5a-zA-Z0-9])', r'\1\2', clean_para)
            # 仅替换段落内的水平空格为单个空格
            clean_para = re.sub(r'[ \t]+', ' ', clean_para).strip()
            # 过滤过短段落
            if len(clean_para) >= self.min_paragraph_len:
                cleaned_paragraphs.append(clean_para)
        # 重新拼接段落（保留\n\n分隔）
        return '\n\n'.join(cleaned_paragraphs)

    def remove_abnormal_formats(self, text):
        """移除异常格式：仅清理段落内的异常，保留换行分隔"""
        abnormal_patterns = [
            r'第\s*\d+\s*页',  # 中文页码：第 5 页
            r'Page\s*\d+',  # 英文页码：Page 10
            r'\d+\s*/\s*\d+',  # 分页标记：1/20
            r'-\s*\d+\s*-',  # 居中页码：- 8 -
            r'^[①②③④⑤⑥⑦⑧⑨⑩]+',  # 特殊序号：① ②
            r'【.*?】|《.*?》',  # 无意义标记：【备注】《标题》
            r'^\s*注：.*?$',  # 注释行：注：本文档...
        ]
        cleaned_text = text
        for pattern in abnormal_patterns:
            # 仅在每行内匹配异常格式（保留换行）
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE)

        # 仅清理段落内的标点/中文间水平空格（不影响换行）
        cleaned_text = re.sub(r'([\u4e00-\u9fa5a-zA-Z0-9])\s+([\.\,\;\:\!\?])', r'\1\2', cleaned_text)
        cleaned_text = re.sub(r'([\.\,\;\:\!\?])\s+([\u4e00-\u9fa5a-zA-Z0-9])', r'\1\2', cleaned_text)
        cleaned_text = re.sub(r'([\u4e00-\u9fa5])[ \t]+([\u4e00-\u9fa5])', r'\1\2', cleaned_text)
        while re.search(r'([\u4e00-\u9fa5])[ \t]+([\u4e00-\u9fa5])', cleaned_text):
            cleaned_text = re.sub(r'([\u4e00-\u9fa5])[ \t]+([\u4e00-\u9fa5])', r'\1\2', cleaned_text)

        # 仅替换段落内的多余水平空格（保留换行）
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
        # 移除每行首尾的水平空格（不影响换行）
        cleaned_text = re.sub(r'^[ \t]+|[ \t]+$', '', cleaned_text, flags=re.MULTILINE)
        return cleaned_text

    def clean(self, text):
        """执行完整清洗流程：保留段落换行分隔"""
        text = self.remove_missing_values(text)
        text = self.remove_abnormal_formats(text)
        return text.strip()


# ===== 3. 词汇处理层（核心修改：修复预分词器组合方式）=====
class VocabProcessingLayer:
    """词汇处理层：子词分词（ByteLevelBPE）+ R代码块精准识别"""

    def __init__(self, vocab_size=5000, save_path="saved_model"):
        # 子词分词器配置
        self.vocab_size = vocab_size
        self.save_path = save_path
        # 特殊符号（与原有BookVocab对齐）
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "<-", "(", ")", ",", "[", "]"]
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        # 初始化子词分词器
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

        # ========== 核心修复：预分词器组合方式 ==========
        from tokenizers.pre_tokenizers import Sequence, Whitespace, Punctuation
        # 正确方式：用Sequence组合多个预分词器（先按空格分，再按标点分）
        self.tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation()])

        # 后处理器：自动添加<SOS>/<EOS>
        self.tokenizer.post_processor = TemplateProcessing(
            single="<SOS> $A <EOS>",
            special_tokens=[("<SOS>", self.sos_idx), ("<EOS>", self.eos_idx)]
        )

    def train_subword_vocab(self, text):
        """基于清洗后的文本训练子词词表（保留换行）"""
        # 拆分文本为行，用于训练（保留段落结构）
        texts = [line.strip() for line in text.split('\n') if line.strip()]
        # 训练子词分词器
        self.tokenizer.train_from_iterator(texts, trainer=BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=2,  # 最小词频（过滤稀有子词）
            show_progress=True
        ))
        # 保存子词词表
        os.makedirs(self.save_path, exist_ok=True)
        self.tokenizer.save(os.path.join(self.save_path, "subword_tokenizer.json"))
        print(f"✅ 子词词表训练完成，保存至：{self.save_path}/subword_tokenizer.json")

    def load_subword_vocab(self):
        """加载预训练的子词分词器"""
        tokenizer_path = os.path.join(self.save_path, "subword_tokenizer.json")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"子词分词器未找到：{tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        print(f"✅ 加载子词分词器完成")

    def test_tokenize(self, text):
        """测试子词分词效果（替换原有jieba测试）"""
        test_text = "R语言数据挖掘中使用dplyr包和%>%管道符，lm函数建模"
        encoding = self.tokenizer.encode(test_text)
        tokens = encoding.tokens
        indices = encoding.ids
        print(f"\n📝 子词分词效果测试：")
        print(f"测试文本：{test_text}")
        print(f"子词结果：{tokens}")
        print(f"子词索引：{indices}")
        return tokens, indices

    def split_code_and_text(self, text):
        """保留原有R代码块识别逻辑"""
        code_pattern = r"""
            (?:^>.*?$) |  # 以>开头的R命令行（整行）
            (?:\w+\s*<-[^,。，;；]+) |  # R赋值语句（避免匹配普通文本中的<-）
            (?:library\([^)]+\)) |  # library函数
            (?:require\([^)]+\)) |  # require函数
            (?:install\.packages\([^)]+\)) |  # install.packages函数
            (?:[a-zA-Z_]+\([^)]+\)\s*%>%\s*[a-zA-Z_]+\([^)]+\)) |  # 管道符语句
            (?:rnorm|mean|sample|plot|table|seq|rep|gl|matrix|data\.frame)\([^)]+\)  # 常见R函数
        """
        code_blocks = re.findall(code_pattern, text, flags=re.VERBOSE | re.MULTILINE | re.DOTALL)
        code_blocks = [block.strip() for block in code_blocks if block.strip()]
        # 从文本中移除识别出的代码块（保留换行）
        pure_text = text
        for block in code_blocks:
            pure_text = pure_text.replace(block, '')
        return code_blocks, pure_text

    def tokenize(self, text, is_code=False):
        """子词分词（替代原有jieba分词）"""
        # 合并代码和纯文本，统一子词分词
        encoding = self.tokenizer.encode(text)
        tokens = encoding.tokens
        indices = encoding.ids
        # 过滤特殊符号外的空token
        tokens = [t for t in tokens if t and t not in [' ', '\n', '\t', '']]
        return tokens, indices

    def simple_stem(self, tokens):
        """轻量级词干提取（保留原有逻辑，适配子词）"""
        stem_rules = {
            'ing$': '', 'ed$': '', 's$': '', 'es$': '', 'ly$': '',
            'tion$': 't', 'ment$': ''
        }
        stemmed_tokens = []
        for token in tokens:
            if re.match(r'[a-zA-Z]+', token) and len(token) > 3:
                stemmed_token = token.lower()
                for pattern, repl in stem_rules.items():
                    stemmed_token = re.sub(pattern, repl, stemmed_token)
                stemmed_tokens.append(stemmed_token)
            else:
                stemmed_tokens.append(token)
        return stemmed_tokens

    def process(self, text, train_vocab=True):
        """执行完整词汇处理流程（替换原有jieba逻辑）"""
        # 1. 识别R代码块
        code_blocks, pure_text = self.split_code_and_text(text)
        # 2. 训练/加载子词词表
        if train_vocab:
            self.train_subword_vocab(text)
        else:
            self.load_subword_vocab()
        # 3. 测试子词分词效果
        self.test_tokenize(text)
        # 4. 子词分词（纯文本+代码块）
        text_tokens, text_indices = self.tokenize(pure_text)
        code_tokens = []
        code_indices = []
        for code in code_blocks:
            c_tokens, c_indices = self.tokenize(code)
            code_tokens.extend(c_tokens)
            code_indices.extend(c_indices)
        # 5. 合并token并词干提取
        all_tokens = text_tokens + code_tokens
        all_indices = text_indices + code_indices
        all_tokens = self.simple_stem(all_tokens)
        # 6. 过滤空token
        all_tokens = [t for t in all_tokens if t.strip() and t not in [' ', '\n', '\t']]
        return all_tokens, all_indices, code_blocks, pure_text


# ===== 4. 噪声过滤层（适配子词分词，保留原有逻辑）=====
class NoiseFilterLayer:
    """噪声过滤层：停用词移除（中英文）、特殊符号处理"""

    def __init__(self, stopwords_cn_path=None):
        # 英文停用词
        self.en_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
                             'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as'}
        # 中文停用词
        self.cn_stopwords = self._load_cn_stopwords(stopwords_cn_path)
        # 保留的R相关特殊符号
        self.keep_symbols = {'<-', '%>%', '(', ')', '[', ']', ',', '.', '+', '-', '*', '/', '%'}

    def _load_cn_stopwords(self, stopwords_path):
        """加载中文停用词表"""
        default_cn_stopwords = {'的', '了', '和', '是', '在', '有', '就', '不', '人', '都', '一', '个', '上', '也',
                                '很', '到', '说', '要'}
        if not stopwords_path:
            return default_cn_stopwords
        try:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                cn_stopwords = set([line.strip() for line in f if line.strip()])
            return cn_stopwords
        except FileNotFoundError:
            print(f"⚠️  中文停用词表 {stopwords_path} 未找到，使用默认停用词")
            return default_cn_stopwords

    def remove_stopwords(self, tokens):
        """移除停用词（适配子词）"""
        filtered_tokens = []
        for token in tokens:
            # 子词级别过滤：仅过滤纯停用词，不过滤子词组合
            if token in self.en_stopwords or token in self.cn_stopwords:
                continue
            filtered_tokens.append(token)
        return filtered_tokens

    def filter_symbols(self, tokens):
        """过滤特殊符号（适配子词）"""
        filtered_tokens = []
        for token in tokens:
            if re.match(r'^[\W_]+$', token) and token not in self.keep_symbols:
                continue
            filtered_tokens.append(token)
        return filtered_tokens

    def filter(self, tokens):
        """执行完整噪声过滤流程"""
        tokens = self.remove_stopwords(tokens)
        tokens = self.filter_symbols(tokens)
        # 仅过滤长度为1的纯符号，保留中文/英文单字
        filtered_tokens = []
        for t in tokens:
            if len(t) == 0:
                continue
            # 长度为1时：仅过滤纯符号（非中文/非英文），保留中文/英文单字
            if len(t) == 1:
                # 判断是否是中文/英文单字（保留），否则过滤
                if re.match(r'^[\u4e00-\u9fa5a-zA-Z]$', t):
                    filtered_tokens.append(t)
                # 是纯符号且不在保留列表 → 过滤
                elif t not in self.keep_symbols:
                    continue
                # 是保留符号 → 保留
                else:
                    filtered_tokens.append(t)
            # 长度>1 → 直接保留
            else:
                filtered_tokens.append(t)
        return filtered_tokens


# ===== 5. 特征工程层（适配子词，保留原有逻辑）=====
class FeatureEngineeringLayer:
    """特征工程层：TF-IDF向量化、N-gram生成"""

    def __init__(self, ngram_range=(1, 3), max_features=2000):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            tokenizer=lambda x: x,  # 禁用默认分词，使用子词
            preprocessor=lambda x: x,
            lowercase=False
        )

    def generate_ngram(self, tokens):
        """生成N-gram特征（适配子词）"""
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i + n])
                ngrams.append(ngram)
        return ngrams

    def tfidf_vectorize(self, tokens_list):
        """TF-IDF向量化（适配子词）"""
        texts = [' '.join(tokens) for tokens in tokens_list]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        return tfidf_matrix.toarray()


# ===== 6. 核心词表类（适配子词分词）=====
class BookVocab:
    def __init__(self, save_path="saved_model"):
        self.save_path = save_path
        # 加载预训练的子词分词器
        self.tokenizer = Tokenizer.from_file(os.path.join(save_path, "subword_tokenizer.json"))
        # 特殊符号映射（与子词分词器对齐）
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "<-", "(", ")", ",", "[", "]"]
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        # 子词词表映射
        self.word2idx = self.tokenizer.get_vocab()
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def encode(self, tokens, max_len=64):
        """子词编码（替换原有整词编码）"""
        # 合并token为文本，用子词分词器编码
        text = ' '.join(tokens)
        encoding = self.tokenizer.encode(text)
        indices = encoding.ids
        # 截断/补齐到max_len
        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            indices += [self.pad_idx] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices):
        """子词解码"""
        # 过滤PAD符号
        indices = [idx for idx in indices if idx != self.pad_idx]
        return self.tokenizer.decode(indices)

    def build_vocab(self, processed_tokens):
        """兼容原有接口，返回子词词表映射"""
        return self.word2idx, self.idx2word


# ===== 7. 数据集类（适配子词编码）=====
class BookDataset(Dataset):
    def __init__(self, tokens_list, vocab, max_len=64):
        self.vocab = vocab
        self.max_len = max_len
        self.pairs = self._make_train_pairs(tokens_list)

    # 构建训练对（保留原有逻辑，适配子词编码）
    def _make_train_pairs(self, tokens_list):
        pairs = []
        chunk_size = 20
        for i in range(0, len(tokens_list) - chunk_size, chunk_size // 2):
            src_tokens = tokens_list[i:i + chunk_size // 2]
            tgt_tokens = tokens_list[i:i + chunk_size]
            src_enc = self.vocab.encode(src_tokens, self.max_len)
            tgt_enc = self.vocab.encode(tgt_tokens, self.max_len)
            pairs.append((src_enc, tgt_enc))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx][0], self.pairs[idx][1]


# ===== 预处理主函数（适配子词分词，保留段落换行）=====
def preprocess_data(
        book_path="data_all.txt",
        custom_dict_path=None,  # 子词分词无需自定义词典，保留参数兼容
        stopwords_cn_path=None,
        batch_size=8,
        max_len=64,
        vocab_size=5000  # 新增：子词词表大小
):
    print("=" * 80)
    print("📚 R语言文本预处理流程启动（子词级分词+保留段落）")
    print("=" * 80)

    # 1. 文本规范化
    print("\n【1. 文本规范化层 - 示例】")
    norm_layer = TextNormalizationLayer()
    try:
        raw_text = norm_layer.normalize(book_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ 找不到数据集文件：{book_path}，请检查路径！")
    # 打印规范化后文本示例（前2000字符）
    print(f"规范化后文本前2000字符：\n{raw_text[:2000]}...")

    # 2. 数据清洗
    print("\n【2. 数据清洗层 - 示例】")
    clean_layer = DataCleaningLayer()
    cleaned_text = clean_layer.clean(raw_text)
    # 打印清洗后文本示例（前2000字符）
    print(f"清洗后纯文本前2000字符：\n{cleaned_text[:2000]}...")

    # 3. 词汇处理（子词分词）
    print("\n【3. 词汇处理层 - 子词分词示例】")
    vocab_process_layer = VocabProcessingLayer(vocab_size=vocab_size)
    all_tokens, all_indices, code_blocks, pure_text = vocab_process_layer.process(cleaned_text)
    # 打印子词分词结果示例（前10个Token）
    print(f"子词分词后Token前10个：\n{all_tokens[:10]}")
    print(f"子词索引前10个：\n{all_indices[:10]}")
    # 打印识别的代码块示例（前2个）
    print(f"识别的真实R代码块前2个：\n{code_blocks[:2]}")

    # 4. 噪声过滤
    print("\n【4. 噪声过滤层 - 示例】")
    filter_layer = NoiseFilterLayer(stopwords_cn_path)
    filtered_tokens = filter_layer.filter(all_tokens)
    print(f"过滤后子词Token前10个：\n{filtered_tokens[:10]}")
    print(f"📝 词汇处理完成：原始token数={len(all_tokens)} → 过滤后={len(filtered_tokens)}")

    # 5. 特征工程
    print("\n【5. 特征工程层 - 示例】")
    feature_layer = FeatureEngineeringLayer(ngram_range=(1, 3), max_features=2000)
    # N-gram示例
    ngrams = feature_layer.generate_ngram(filtered_tokens)
    print(f"N-gram生成示例（前5个）：\n{ngrams[:5]}")
    print(f"📊 N-gram生成完成：共{len(ngrams)}个（1-3 gram）")

    # 拆分清洗后的文本为独立段落（保留\n\n分隔）
    paragraphs = cleaned_text.split('\n\n')
    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    real_tokens_list = []  # 存储每个段落的过滤后token列表
    for para in paragraphs:
        if para.strip():
            para_tokens, _ = vocab_process_layer.tokenize(para)
            para_tokens = filter_layer.filter(para_tokens)
            if para_tokens:
                real_tokens_list.append(para_tokens)

    # 兜底：如果所有段落过滤后都为空，用原filtered_tokens作为单样本
    if not real_tokens_list:
        real_tokens_list = [filtered_tokens]

    # TF-IDF向量化
    tfidf_matrix = feature_layer.tfidf_vectorize(real_tokens_list)
    print(f"🔍 按段落拆分后的有效样本数：{len(real_tokens_list)}")
    print(f"TF-IDF矩阵示例（前3行×前5列）：\n{tfidf_matrix[:3, :5]}")
    print(f"📊 TF-IDF向量化完成：维度={tfidf_matrix.shape}（样本数×特征数）")

    # 6. 词表构建（子词）
    print("\n【6. 词表构建层 - 子词词表示例】")
    vocab = BookVocab()
    word2idx, idx2word = vocab.build_vocab(filtered_tokens)
    vocab_size = len(word2idx)
    # 打印子词词表示例（前15个映射）
    vocab_sample = dict(list(word2idx.items())[:15])
    print(f"子词词表映射示例（前15个）：\n{vocab_sample}")
    print(f"✅ 子词词表构建完成：词表大小={vocab_size}")

    # 7. 数据集编码
    print("\n【7. 数据集编码层 - 示例】")
    dataset = BookDataset(filtered_tokens, vocab, max_len)
    # 打印第一个训练对的编码示例
    if len(dataset) > 0:
        first_pair = dataset[0]
        src_enc, tgt_enc = first_pair
        print(f"第一个训练对 - 输入编码前10个索引：\n{src_enc[:10].tolist()}")
        print(f"第一个训练对 - 目标编码前10个索引：\n{tgt_enc[:10].tolist()}")
        # 解码示例
        src_dec = vocab.decode(src_enc[:10].tolist())
        print(f"输入编码解码示例：\n{src_dec}")
    print(f"✅ 数据集构建完成：训练样本数={len(dataset)}")

    # 8. 构建DataLoader
    print("\n【8. 数据加载器 - 示例】")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,  # 关闭CPU环境的pin_memory警告
        drop_last=True
    )
    # 打印批次形状示例
    if len(dataloader) > 0:
        first_batch = next(iter(dataloader))
        src_batch, tgt_batch = first_batch
        print(f"第一个批次形状：输入={src_batch.shape}，目标={tgt_batch.shape}")
    print(f"✅ 数据加载器构建完成：批次大小={batch_size}")

    print("\n" + "=" * 80)
    print("📌 最终预处理结果汇总（子词级+保留段落）")
    print("=" * 80)
    print(f"子词词表大小：{vocab_size}")
    print(f"过滤后子词Token示例：{filtered_tokens[:10]}")
    print(f"TF-IDF真实样本数：{len(real_tokens_list)}")
    if len(dataloader) > 0:
        print(f"第一个训练样本形状：{src_batch.shape}")

    return dataloader, vocab, vocab_size, filtered_tokens, tfidf_matrix


# ===== 测试入口 =====
if __name__ == "__main__":
    try:
        dataloader, vocab, vocab_size, tokens, tfidf = preprocess_data(
            book_path="data_all.txt",  # 替换为你的文本文件路径
            vocab_size=5000
        )
        print("\n🎉 预处理流程执行完成（保留段落分隔）！")
    except Exception as e:
        print(f"\n❌ 预处理流程执行失败：{e}")