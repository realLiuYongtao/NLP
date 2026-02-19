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


show_word_tokens = tokenize(lines, 'word')
print("测试 word tokens")
print(show_word_tokens[0:10])
show_char_tokens = tokenize(lines, 'char')
print("测试 char tokens")
print(show_char_tokens[0:10])

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


print("测试 count corpus:")
print(count_corpus(lines))

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

print("测试 vocab:")

vocab.print_vocab(n=100)
print("now ->", vocab['now'])
print("unknown ->", vocab['xyz'])
print("sentence ->", vocab[['<bos>', 'dear', 'gatsby', '<eos>']])

