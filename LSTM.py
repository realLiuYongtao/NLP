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

################################################
# 构建 LSTM前向传播和参数初始化函数
################################################
"""初始化RNN模型参数并返回参数列表"""
def get_lstm_params(vocab_size, num_hiddens, device):
    """
    初始化 LSTM 模型参数（手写版）
    - 使用 one-hot 编码，因此输入维度 = 词表大小
    """

    # LSTM 的输入维度等于输出维度（one-hot）
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        """小尺度正态分布初始化权重"""
        return torch.randn(size=shape, device=device) * 0.01  # *0.01在标准正态分布基础上缩小

    def two_W_one_b():
        """
        初始化 LSTM 的三个关键参数：
        1. W_x：输入到隐藏状态的权重
        2. W_h：隐藏状态到隐藏状态的权重
        3. b  ：偏置项（初始化为 0）

        维度规则：
        - W_x: (num_inputs, num_hiddens)
        - W_h: (num_hiddens, num_hiddens)
        - b  : (num_hiddens,)
        """
        return (
            normal((num_inputs, num_hiddens)),  # W_x
            normal((num_hiddens, num_hiddens)),  # W_h
            torch.zeros(num_hiddens, device=device)  # b
        )

    # -------- LSTM 四组门控参数 --------
    W_xi, W_hi, b_i = two_W_one_b()  # 输入门 (Input Gate)
    W_xf, W_hf, b_f = two_W_one_b()  # 遗忘门 (Forget Gate)
    W_xo, W_ho, b_o = two_W_one_b()  # 输出门 (Output Gate)
    W_xc, W_hc, b_c = two_W_one_b()  # 候选记忆单元 (Candidate Cell)

    # -------- 输出层参数 --------
    W_hq = normal((num_hiddens, num_outputs))  # 隐藏层 → 输出层
    b_q = torch.zeros(num_outputs, device=device)  # 输出层偏置

    # 将所有参数打包，便于优化
    params = [
        W_xi, W_hi, b_i,  # 输入门
        W_xf, W_hf, b_f,  # 遗忘门
        W_xo, W_ho, b_o,  # 输出门
        W_xc, W_hc, b_c,  # 候选记忆
        W_hq, b_q  # 输出层
    ]

    # 开启梯度
    for param in params:
        param.requires_grad_(True)

    return params


"""初始化RNN隐藏状态"""
def init_lstm_state(batch_size, num_hiddens, device):
    # 长期记忆 C 和短期记忆 H 都需要初始化
    return (
        torch.zeros((batch_size, num_hiddens), device=device),  # 长期记忆 C
        torch.zeros((batch_size, num_hiddens), device=device)  # 短期记忆 H
    )


"""实现前向传播"""
def lstm(inputs, state, params):
    [
        W_xi, W_hi, b_i,
        W_xf, W_hf, b_f,
        W_xo, W_ho, b_o,
        W_xc, W_hc, b_c,
        W_hq, b_q
    ] = params  # get_lstm_params

    H, C = state  # init_lstm_state 中间状态

    outputs = []

    # 这里的循环是在时间步上的循环
    # 在将最初输入的数据维度是(batch_size,seq_len,vocab_size)
    # 在进行前向传播前，对input进行了转置->(seq_len,batch_size,vocab_size)
    for X in inputs:
        # 1. 输入门
        # I_t = sigmoid(W_xi * X_t + W_hi * H_{t-1} + b_i)
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)

        # 2. 遗忘门
        # F_t = sigmoid(W_xf * X_t + W_hf * H_{t-1} + b_f)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)

        # 3. 输出门
        # O_t = sigmoid(W_xo * X_t + W_ho * H_{t-1} + b_o)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)

        # 4. 候选记忆单元
        # C_tilde = tanh(W_xc * X_t + W_hc * H_{t-1} + b_c)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)

        # 5. 更新细胞状态
        # C_t = F_t * C_{t-1} + I_t * C_tilde
        C = F * C + I * C_tilda

        # 6. 更新隐状态
        # H_t = O_t * tanh(C_t)
        H = O * torch.tanh(C)

        # 7. 计算输出
        # Y_t = W_hq * H_t + b_q
        Y = (H @ W_hq) + b_q # Y.shape = (batch_size,vocab_size)

        outputs.append(Y) # outputs = list(seq_len个(batch_size,vocab_size))

    # pdb.set_trace()
    # 拼接所有时间步的输出
    # 在seq_len维度上对outputs合并,(seq_len,batch_size,vocab_size)
    return torch.cat(outputs, dim=0), (H, C)


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

        # !!!注意！！！常规call函数默认调用self.forward_fn，这里重构call函数
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

    for X, Y in train_iter: # 输入/标签
        # 第一个batch：初始化 state
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 其它batch：detach，避免计算图越来越长导致“卡死”
            for s in state:
                s.detach_()

        # 便于计算损失
        y = Y.T.reshape(-1)  # (B, T) -> (T*B,)
        # 输入和标签加入设备
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

    """
    每个epoch看到的数据顺序是相同的（因为是顺序切分）
    每个epoch处理完所有数据（遍历train_iter中的所有批次）
    总共训练500次整个数据集
    """
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
# 训练！！！
################################################
vocab_size, num_hiddens, device = len(vocab), 256, "cpu"
num_epochs, lr = 500, 0.01

model = RNNModel(
    len(vocab),
    num_hiddens,
    device,
    get_lstm_params,
    init_lstm_state,
    lstm
)

print('='*30)
print("===============开始训练")
print('='*30)
train(model, train_iter, vocab, lr, num_epochs, device)