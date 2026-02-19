import collections
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
    :return: 由 (X, Y) 元组组成的列表，X/Y分布为输入/输出序列 形状[batch_size, num_steps]，一个元组为一个训练批次

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

################################################
# 构建 RNN前向传播和参数初始化函数
################################################
# 初始化以下5个参数：
# 1. W_xh: 输入到隐藏层的权重矩阵
# 2. W_hh: 隐藏层到隐藏层(得到下一时刻的隐藏层)的权重矩阵
# 3. b_h:  隐藏层的偏置向量
# 4. W_hq: 隐藏层到输出层的权重矩阵
# 5. b_q:  输出层的偏置向量
"""初始化RNN模型参数并返回参数列表"""
def get_params(vocab_size, num_hiddens, device):
    """
    初始化循环神经网络（RNN）模型的参数。
    参数：
        vocab_size (int): 词汇表大小，即输入和输出的特征维度。
        num_hiddens (int): 隐藏层的神经元数量（隐藏单元数）。
        device (str or torch.device): 设备（如 'cpu' 或 'cuda'），参数将在此设备上初始化。
    返回：
        params (list): 包含模型参数的列表，每个参数都已启用梯度计算。
    """

    # 这一步是因为后续对输入token使用了独热编码，所以每一个token的维度为vocab_size
    # 一次输入的数据形状为(seq_len,batch_size,vocab_size)
    # seq_len代表时间步，在这个维度上循环
    # 根据深度学习的特性，每次并行处理batch_size个数据
    # 每个时间步的输入数据形状为(batch_size, vocab_size)
    # 其中每个样本用vocab_size维的向量表示（如one-hot）
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        """
        使用正态分布：torch.randn 生成均值为0，标准差为1的正态分布
        缩放因子 0.01：防止初始值过大，避免梯度爆炸
        理论依据：Xavier/Glorot初始化的简化版，适合tanh激活函数
        """
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    # h_t = tanh(X_t @ W_xh + h_{t-1} @ W_hh + b_h)  隐藏状态更新
    W_xh = normal((num_inputs, num_hiddens))  # 在normal函数中设置了设备
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)

    # 输出层参数
    # O_t = h_t @ W_hq + b_q  输出计算
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        # 启用自动微分：所有参数都需要计算梯度
        # 训练准备：为反向传播做准备
        param.requires_grad_(True)

    return params


"""初始化RNN隐藏状态"""
def init_rnn_state(batch_size, num_hiddens, device):
    """
    初始化 RNN 的隐藏状态。
    参数：
        batch_size (int): 批量大小，即一次处理的序列数量。
        num_hiddens (int): 隐藏层的单元数（隐藏状态的大小）。

    返回：
        tuple: 包含初始化的隐藏状态张量，形状为 (batch_size, num_hiddens)。
    """
    # 这里要用 device=device
    # 保证一致性：确保张量与模型/数据在同一设备
    # 注意结尾的,!!!返回的是一个元组，为了与多层RNN预留接口保存一致性！！！
    # 单层RNN：返回(h,)
    # 多层RNN：返回(h1, h2, h3, ...)
    return (torch.zeros((batch_size, num_hiddens), device=device),)


"""实现前向传播"""
def rnn(inputs, state, params):
    """
    实现循环神经网络（RNN）的前向传播。
    参数：
        inputs (tensor): 输入数据，形状为 (时间步数量, 批量大小, 词表大小)。
        state (tuple): RNN 的初始状态，形状为 (批量大小, 隐藏单元数)。
        params (list): RNN 的参数，包括 W_xh, W_hh, b_h, W_hq, b_q。

    返回：
        outputs (tensor): 输出结果，形状为 (时间步数量 * 批量大小, 词表大小)。
        new_state (tuple): 更新后的隐藏状态。
    """
    W_xh, W_hh, b_h, W_hq, b_q = params  # 传入参数
    H, = state  # 获取初始隐藏状态
    outputs = []  # 存储所有时间步的输出

    # 遍历每个时间步的输入 inputs (时间步数量, 批量大小, 词表大小) X (批量大小, 词表大小)
    # 数据流说明：
    # 外层循环遍历seq_len个时间步
    # 每个时间步X的形状为(batch_size, vocab_size)
    # 批次内的batch_size个序列并行计算
    # 每个序列的当前token用vocab_size维向量表示
    for X in inputs:
        # 计算隐藏状态 H  h_t = tanh(X_t @ W_xh + h_{t-1} @ W_hh + b_h)  隐藏状态更新
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        # 计算输出 Y  O_t = h_t @ W_hq + b_q  输出计算
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)  # 记录当前时间步的输出

    # 将所有时间步的输出拼接在一起
    # 假设 seq_len = 10, batch_size = 32
    # 每个时间步 Y 形状: (32, 10000) 因为这里使用独热编码 即 (batch_size, vocab_size)
    # ← 这里的vocab_size是one-hot维度  表示：每个批次样本 × 对词表中每个词的预测分数
    # outputs 列表: [Y1, Y2, ..., Y10]，每个都是(32, 10000)
    # 拼接后: (320, 10000)
    # 同时返回最后一个隐藏状态: (32, 256)
    # 注意这里，隐藏状态返回的是一个元组，为了与多层RNN返回多个中间状态预留接口保存一致性！！！
    return torch.cat(outputs, dim=0), (H,)


################################################
# 从零构建 通用的RNN模型 RNN/LSTM/GRU/DEEP_RNN通用
################################################
class RNNModel:
    """
    这是一个通用的RNN类，
    可自定义前向传播forward_fn和参数初始化函数init_state以及初始化状态函数
    以实现经典RNN，LSTM，GRU等等
    """

    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        """
        初始化 RNN 模型！！！
        参数：
            vocab_size (int): 词汇表大小，即输入和输出的特征数量，因为用的one-hot编码
            num_hiddens (int): 隐藏单元数量，决定 RNN 的记忆容量。
            device (torch.device): 计算设备。
            get_params (function): 获取模型参数的函数。
            init_state (function): 初始化隐藏状态的函数。
            forward_fn (function): RNN 的前向传播函数。
        """
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        # 调用 get_params 初始化权重和偏置
        self.params = get_params(vocab_size, num_hiddens, device)
        # 记录初始化状态函数和前向传播函数
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        """
        执行模型的前向传播。
        参数：
            X (tensor): 输入数据，形状（批量大小，序列长度）。
            state (tuple): 隐藏状态。

        返回：
            outputs (tensor): 预测结果，形状（时间步数量 * 批量大小，词表大小）。
            new_state (tuple): 更新后的隐藏状态。
        """
        # 对输入 X 进行 one-hot 编码并转换为 float32
        # 形状：(序列长度，批量大小，词表大小)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        """
        初始化隐藏状态。
        参数：
            batch_size (int): 批量大小。
            device (torch.device): 计算设备。

        返回：
            tuple: 初始化的隐藏状态。
        """
        return self.init_state(batch_size, self.num_hiddens, device)


################################################
# 实例化RNN模型并测试
################################################
# 设置隐藏层神经元数量
num_hiddens = 512

# 创建输入数据 X，形状（批量大小，序列长度）
X = torch.arange(10).reshape((2, 5))

# 设置divice
device = 'cpu'

# 创建 RNN 模型实例
net = RNNModel(
    len(vocab),
    num_hiddens,
    device,
    get_params,
    init_rnn_state,
    rnn
)

# 初始化隐藏状态
state = net.begin_state(X.shape[0], device)

# 执行前向传播，获得预测结果 Y 和新的隐藏状态 new_state
Y, new_state = net(X.to(device), state)

# 输出预测结果形状、隐藏状态的长度以及隐藏状态的形状
print("测试网络实例：")
print("输出形状(batch_size*seq_len,vocab_size):", Y.shape)
print("隐藏状态：", new_state)
print("隐藏状态长度：", len(new_state))
print("隐藏状态的形状：(batch_size,num_hiddens)", new_state[0].shape)

################################################
# 工具函数 RNN/LSTM/GRU/DEEP_RNN通用
################################################
""" 预测函数 """
def predict(prefix, num_preds, net, vocab, device):
    """
    在给定的前缀字符串之后，使用 RNN 模型生成新的字符序列。
    参数：
        prefix (str): 生成序列的起始字符串（种子文本）。
        num_preds (int): 需要生成的字符数。
        net (RNNModelScratch): 训练好的循环神经网络模型。
        vocab (Vocab): 词汇表，提供字符与索引的映射关系。
        device (torch.device): 计算设备（'cpu' 或 'cuda'）。

    返回：
        str: 生成的完整文本（包含前缀和预测的新字符）。
    """
    # 初始化 RNN 的隐藏状态，batch_size=1 处理单个序列
    state = net.begin_state(batch_size=1, device=device)

    # 将 prefix 的第一个字符转换为索引并存入输出列表
    outputs = [vocab[prefix[0]]]

    # 定义一个 lambda 函数，获取当前最后一个字符的索引并转换为模型输入
    get_input = lambda: torch.tensor(
        [outputs[-1]], device=device
    ).reshape((1, 1))

    # 预热期：将 prefix 剩余字符依次输入网络，帮助 RNN 进入适当的状态
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])

    # 生成 num_preds 个新的字符
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))

    # 将输出索引列表转换回字符，并连接成字符串
    return ''.join([vocab.idx_to_token[i] for i in outputs])


print("测试predict：")
print(predict('dear gatsby ', 10, net, vocab, device))


class Accumulator:
    """
    在多个变量上进行累加的工具类
    """

    def __init__(self, n):
        """
        参数:
            n (int): 需要累加的变量个数
        """
        self.data = [0.0] * n

    def add(self, *args):
        """
        将传入的值逐项累加
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        """
        清零
        """
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        """
        允许用 metric[i] 的方式访问
        """
        return self.data[idx]


# 梯度裁剪
def grad_clipping(net, theta):
    """
    梯度裁剪，防止梯度爆炸

    参数:
        net: 模型（nn.Module 或自定义 RNN）
        theta (float): 梯度范数阈值
    """
    if isinstance(net, torch.nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params  # RNN 的参数列表
    # 计算梯度的 L2 范数
    norm = torch.sqrt(
        sum(torch.sum(p.grad ** 2) for p in params if p.grad is not None)
    )
    # 若超过阈值，则按比例缩放
    if norm > theta:
        for p in params:
            if p.grad is not None:
                p.grad[:] *= theta / norm


"""训练遍历一遍数据集"""
def train_epoch(net, train_iter, loss, optimizer, device, use_random_iter):
    state = None
    metric = Accumulator(2)  # [total_loss, total_tokens]

    for X, Y in train_iter:
        # 第一个batch：初始化 state
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 其它batch：detach，避免计算图越来越长导致“卡死”
            for s in state:
                s.detach_()

        y = Y.T.reshape(-1)  # (B, T) -> (T*B,)
        X, y = X.to(device), y.to(device)

        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()

        optimizer.zero_grad()
        l.backward()
        grad_clipping(net, 1.0)
        optimizer.step()

        metric.add(l * y.numel(), y.numel())

    return math.exp(metric[0] / metric[1])


def train(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.params, lr=lr)

    epochs, ppls = [], []

    for epoch in range(num_epochs):
        ppl = train_epoch(net, train_iter, loss, optimizer, device, use_random_iter)

        print(f"[epoch {epoch + 1:4d}] perplexity = {ppl:.2f}")

        epochs.append(epoch + 1)
        ppls.append(ppl)

        if (epoch + 1) % 5 == 0:
            print(predict('dear gatsby ', 50, net, vocab, device))

    plt.figure(figsize=(6, 3))
    plt.plot(epochs, ppls)
    plt.xlabel('epoch')
    plt.ylabel('perplexity')
    plt.grid(True)
    plt.show()

    print("\nFinal sample:")
    print(predict('dear gatsby ', 50, net, vocab, device))


################################################
# 训练！
################################################
num_epochs = 50
lr = 1e-3

print('='*60)
print("===============开始训练")
print('='*60)

train(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False)
