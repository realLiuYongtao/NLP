import collections
import pdb
import re
import math
import time
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

################################################
# 读取并清洗文本 RNN/LSTM/GRU/DEEP_RNN通用
################################################
"""
step1 读取并清洗文本
"""
file_path = './novel.txt'


# 读取文本并预处理
def read_txt_file(file_path):
    # 打开文本
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 对每一行处理
    cleaned_lines = [
        re.sub('[^A-Za-z]+', ' ', line).strip().lower()
        for line in lines
    ]
    """
    re.sub('[^A-Za-z]+', ' ', line)：
    使用正则表达式将非字母字符（包括数字、标点、特殊符号等）替换为空格
    .strip()：移除行首尾的空白字符
    .lower()：将所有字母转换为小写
    """
    # 对每一行文本进行清洗处理，只保留英文字母，并将所有字符转换为小写。
    # 文本预处理流程中的常见步骤，用于将原始文本转换为更适合后续分析的标准化格式。
    return cleaned_lines


# 读取并预处理后的文本
lines = read_txt_file(file_path)

################################################
# Tokenize RNN/LSTM/GRU/DEEP_RNN通用
################################################
"""
step2 构建tokenize

将原始文本分割成更小单元（tokens）的过程，
这些单元是语言模型能够理解和处理的基本元素。
典型的：
1.分为单词word
2.分为字母字符char
3.分词为词缀（大模型使用）
"""
def tokenize(lines, token='word'):
    # 对两种构建tokenize的演示
    # 此段代码无作用
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        raise ValueError(token)


################################################
# 构建词表 Vocab
################################################
"""
step3 统计tokens的词频
"""
def count_corpus(corpus):
    """
    统计语料中每个 token 出现的次数
    tokens:
        - 可以是 ['a', 'b', 'c']
        - 也可以是 [['a','b'], ['c','d']]
    返回：
        Counter({'a': 3, 'b': 2, ...})
    """

    all_tokens = []
    # 处理情况：corpus 是文本行列表（每行是字符串）
    if len(corpus) > 0 and isinstance(corpus[0], str):
        # 字符串列表，每行可能包含多个单词
        for line in corpus:
            # 按空格分割单词
            words = line.split()
            all_tokens.extend(words)  # 直接添加分割后的单词列表

    # 处理情况：corpus 已经是 token 列表（一维列表）
    elif len(corpus) > 0 and isinstance(corpus[0], (str, int, float)):
        all_tokens = list(corpus)

    # 处理情况：corpus 是 token 的嵌套列表（二维列表）
    elif len(corpus) > 0 and isinstance(corpus[0], list):
        for sublist in corpus:
            all_tokens.extend(sublist)

    print(len(all_tokens))
    return collections.Counter(all_tokens)


"""
构建词表 Vocab
"""
class Vocab:
    def __init__(self, tokens=None):
        """
        构建词表
        tokens: token 列表（可以是一维或二维）
        """
        if tokens is None:
            tokens = []

        # 1. 统计词频
        counter = count_corpus(tokens)

        # 2. 初始化特殊符号
        self.idx_to_token = [' ', '<unk>', '<bos>', '<eos>']  # 由索引找到token
        self.token_to_idx = {  # 由token找到索引
            ' ': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
        }

        # 3. 按频率从高到低加入普通 token
        # 频率越高，越早加入，对应索引越小
        for token, freq in counter.most_common():
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        # 长度
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        # 将 token（文本单元）转换为对应的索引编号

        # 单个 token
        if not isinstance(tokens, (list, tuple)):  # 如果不是列表或元组，认为是单个 token
            # 如果 token 不在词汇表中，返回 <unk>（未知词）的索引
            return self.token_to_idx.get(tokens, self.token_to_idx['<unk>'])

        # token 列表 (多个 token)
        indices = []
        for token in tokens:
            # self[token] 调用同一个 __getitem__ 方法
            indices.append(self[token])  # 递归调用自身
        return indices

    def print_vocab(self, n=10):
        # 打印词表前n位
        print("===== Vocabulary Preview =====")
        print("index -> token")
        for i in range(min(n, len(self.idx_to_token))):
            print(f"{i:>3} -> {self.idx_to_token[i]}")


"""
构建字符级corpus 语料库
"""
vocab = Vocab(lines)


################################################
# 构建 dataloader RNN/LSTM/GRU/DEEP_RNN通用
################################################
def build_corpus_ids(lines, vocab):
    """
    :param lines: 预处理好的文本，每行可能包含多个单词
    :param vocab: 构建好的词表字典，格式为 {word: id}
    :return: 将文本按word加入一个列表，并按照词表映射成数字ID，再转换成张量
    """
    # 第一步，将文本全部去掉空格和段落，转化成单词的列表保存
    words = []
    for line in lines:
        if line.strip():  # 过滤掉空行（使用 strip() 检查）
            words += line.split()  # 将每行按空格分割成单词，所有单词合并到一个列表中
    print("测试build_corpus_ids:")
    print(words[:10])
    # 第二步，创建PyTorch的长整型张量，使用词表 vocab 将单词映射为对应的数字ID
    result = torch.tensor([vocab[w] for w in words], dtype=torch.long)
    print(result[:10])
    return result


# 生成顺序训练的批次数据
def train_iter_sequential_simple(corpus_ids, batch_size, num_steps, device='cpu'):
    """
    :param corpus_ids: 转换成张量后的corpus
    :param batch_size: 批次大小
    :param num_steps: 每个训练样本的序列长度（时间步数）
    :param device: 训练设备
    :return: 由 (X, Y) 元组组成的列表，X/Y：输入/输出序列 形状[batch_size, num_steps]，一个元组为一个训练批次
    """
    corpus_ids = corpus_ids.to(device)
    # 获取总token数
    N = corpus_ids.numel()
    # 验证语料长度足够，保证后续能顺利进行
    assert N > batch_size * (num_steps + 1), "语料太短，batch_size*num_steps 太大了"

    """
    我们希望一次输入一个batch_size的数据并行训练，所以Xs与Ys的长度要被batch_size整除
    假设 Xs = corpus_ids[:num] Ys = corpus_ids[1:num+1]（右移一位）
    Ys是Xs右移一位，所以假设num是能被batch_size整除的最大整数，实际上需要num+1个token
    """
    # 截断到能整除 batch_size 的最大整数
    n = ((N - 1) // batch_size) * batch_size

    # 输入序列
    Xs = corpus_ids[:n]  # 取0~n-1
    Xs = Xs.reshape(batch_size, -1)  # reshape成batch_size行
    # 输入序列右移一位成输出序列
    Ys = corpus_ids[1:n + 1]  # 取1~n
    Ys = Ys.reshape(batch_size, -1)

    # 创建一个空列表，用于存储所有训练批次
    # 每个批次是一个 (X, Y) 元组
    batches = []  # ★ 所有 batch 放在这里

    L = Xs.shape[1]
    for t in range(0, L - num_steps + 1, num_steps):  # num_steps 滑动窗口移动步长
        # : 表示取所有批次（batch_size 维度）
        # t:t + num_steps 表示取连续的 num_steps 个时间步
        # 形状从 [batch_size, L] 变为 [batch_size, num_steps]
        X = Xs[:, t:t + num_steps]
        Y = Ys[:, t:t + num_steps]
        batches.append((X, Y))  # ★ 收集起来

    return batches


corpus_ids = build_corpus_ids(lines, vocab)
print("测试dataloader：")
print("Total tokens:", corpus_ids.shape)

batch_size = 32
num_steps = 35

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

train_iter = train_iter_sequential_simple(
    corpus_ids, batch_size, num_steps, device
)

print("测试train_iter:")
for X, Y in train_iter:
    print("X shape:", X.shape)  # (batch_size, num_steps)
    print("Y shape:", Y.shape)
    print("X[0]:", X[0])
    print("Y[0]:", Y[0])
    break


