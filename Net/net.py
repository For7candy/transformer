import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn   # 基于plt包的数据可视化包
seaborn.set_context(context="talk")

from tools.clones import clones


class Embeddings(nn.Module):
    """
    通过初始化参数d_model and vocab 初始化一个vocab*d_model大小的词表lut，
    forward()函数将输入形式为one-hot编码的x，通过查找词表lut转换为Embedding
    **其中lut是需要训练的**
    """
    def __init__(self, d_model, vocab):
        # d_model为词向量维度(512)，vocab为词表大小，vocab*d_model组成我们需要训练的矩阵lut
        super(Embeddings, self).__init__()
        # nn.Embedding是pytorch内的查找表，用于存储固定词典和大小的嵌入
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x为输入，(batch_size, sequence.length, one-hot)   # !!!这里知乎答主应该理解有错误
        # one-hot转词嵌入
        # (batch_size, sequence.length, 512)
        return self.lut(x) * math.sqrt(self.d_model)  # 【?】
        # return self.lut(x)


class PositionalEncoding(nn.Module):
    """
    位置编码,没有需要训练的参数
    """
    def __init__(self, d_model, dropout, max_len=5000):
        # max_len 表示准备好5000个位置编码向量,实际上100-200即可
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # (5000*512)
        pe = torch.zeros(max_len, d_model)
        # (5000)-> (5000,1), 用arange()生成使它本身自带顺序属性
        position = torch.arange(0, max_len).unsqueeze(1)

        # 创造一个衰减序列，这个序列是从e^0开始，向x负半轴取值，因此是衰减的
        # 序列长度为d_model/2，因为后续是分sin()和cos()进行位置编码
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                    -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (5000,512)->(1,5000,512),为batch_size占据位置
        pe = pe.unsqueeze(0)
        # Module.register_buffer(name, tensor)
        # 将tensor作为模型的一部分，但是不会进行反向传播和参数更新
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    # 注意力机制：Q来自target sequence， K和V来自source sequence;自注意力机制：Q、K、V都来自target sequence
    # query, key, value的形状类似于(30, 8, 10, 64), (30, 8, 11, 64),
    # (30, 8, 11, 64)，例如30是batch.size，即当前batch中有多少一个序列；
    # 8=head.num，注意力头的个数；
    # 10=目标序列中词的个数，64是每个词对应的向量表示；
    # 11=源语言序列传过来的memory中，当前序列的词的个数，
    # 64是每个词对应的向量表示。
    d_k = query.size(-1)  # d_k=64
    # (30,8,10,11)
    scores = torch.matmul(query, key.transpose(-2, -1) /
            math.sqrt(d_k))

    # 使用mask，对已经计算好的scores，按照mask矩阵，填-1e9，
    # 然后在下一步计算softmax的时候，被设置成-1e9的数对应的值~0,被忽视
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 对scores的最后一个维度进行softmax【？这个softmax是如何运行的】
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)  # 执行一次dropout

    # 返回的第一项，是(30,8,10, 11)乘以（最后两个维度相乘）
    # value=(30,8,11,64)，得到的tensor是(30,8,10,64)，
    # 和query的最初的形状一样。另外，返回p_attn，形状为(30,8,10,11).
    # 注意，这里返回p_attn主要是用来可视化显示多头注意力机制。
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        # h:注意力头数；d_model:词向量维度；
        super(MultiHeadedAttention, self).__init__()

        assert d_model % h == 0  # 【?】
        self.d_k = d_model // h  # 512//8=64
        self.h = h
        # 定义四个Linear networks, 每个的大小是(512, 512)的，
        # 每个Linear network里面有两类可训练参数，Weights，
        # 其大小为512*512，以及biases，其大小为512=d_model。
        self.linears = clones(nn.Linear(d_model, d_model), 4)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 注意，输入query的形状类似于(30, 10, 512)，
        # key.size() ~ (30, 11, 512),
        # 以及value.size() ~ (30, 11, 512)

        if mask is not None:  # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # mask下回细细分解。
        nbatches = query.size(0)  # e.g., nbatches=30
        # 1) Do all the linear projections in batch from
        # d_model => h x d_k
        # 这里只对应前三个Linear，前面是4个没错
        # for l, x in zip(self.linears, (query, key, value)) -> 首先将三个矩阵分别对应一个linear
        # l(x):矩阵经过一个linear
        # .view(nbatches, -1, self.h, self.d_k):线性变换后的矩阵会进行 展平 (batch_size, -1, 注意力头数量， 注意力头的特征维度）
        #                                                              (30, 10, 8, 64)
        # .transpose(1, 2)      (30, 8, 10, 64)
        # (30, 10, 512) -> (30, 8, 10, 64) 相当于是说Q,K,V的单位关注点从一个词向量拆分为更小的语义向量，此时的语义向量没有实际对应的词，只是作为“中间语义”
        # 【?】
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k)
                             .transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        # 调用上面定义好的attention函数，输出的x形状为(30, 8, 10, 64)；！！！如果是自注意力的话应该是(30, 8, 10, 10),表征了输入各词向量彼此的关系
        # attn的形状为(30, 8, 10=target.seq.len, 11=src.seq.len)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # x ~ (30, 8, 10, 64) -> transpose(1,2) ->
        # (30, 10, 8, 64) -> contiguous() and view ->
        # (30, 10, 8*64) = (30, 10, 512)
        # 自注意力应该为：
        # x ~ (30, 8, 10, 10) -> transpose(1, 2)->
        # (30, 10, 8, 10)-> contiguous() and view() ->
        # (30, 10, 8*10) -> 相当于把8个注意力头进行了融合 【?】

        return self.linears[-1](x)
        # 执行第四个Linear network，把(30, 10, 512)经过一次linear network，
        # 得到(30, 10, 512).


class LayerNorm(nn.Module):
    """
    归一化层
    """
    def __init__(self, features, eps=1e-6):
        # features=d_model=512, eps=epsilon 用于分母的非0化平滑
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        # a_2 是一个可训练参数向量，(512)
        self.b_2 = nn.Parameter(torch.zeros(features))
        # b_2 也是一个可训练参数向量, (512)
        self.eps = eps

    def forward(self, x):
        # x 的形状为(batch.size, sequence.len, 512)
        mean = x.mean(-1, keepdim=True)
        # 对x的最后一个维度，取平均值，得到tensor (batch.size, seq.len, 1)
        std = x.std(-1, keepdim=True)
        # 对x的最后一个维度，取标准方差，得(batch.size, seq.len, 1)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        # 本质上类似于（x-mean)/std，不过这里加入了两个可训练向量
        # a_2 and b_2，以及分母上增加一个极小值epsilon，用来防止std为0的时候的除法溢出


class SublayerConnection(nn.Module):
    """
    残差层后接一个归一化层
    此处为了代码简洁将归一化层放在了前面
    """
    def __init__(self, size, dropout):
        # size=d_model=512; dropout=0.1
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)  # (512)，用来定义a_2和b_2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        将残差层用于任一sublayer，只要求它的size相同，即处理前后的维度相同
        :param x: 输入
        :param sublayer: 需要应用的层
        :return: x+sublayer(x) -> 残差结构
        """
        # 归一化->sublayer->dropout->+x
        # return x+self.dropout(self.norm(sublayer(x)))
        return x+self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        # d_model = 512
        # d_ff = 2048 = 512*4
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        # 构建第一个全连接层，(512, 2048)，其中有两种可训练参数：
        # weights矩阵，(512, 2048)，以及
        # biases偏移向量, (2048)
        self.w_2 = nn.Linear(d_ff, d_model)
        # 构建第二个全连接层, (2048, 512)，两种可训练参数：
        # weights矩阵，(2048, 512)，以及
        # biases偏移向量, (512)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape = (batch.size, sequence.len, 512)
        # 例如, (30, 10, 512)
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        # x (30, 10, 512) -> self.w_1 -> (30, 10, 2048)
        # -> relu -> (30, 10, 2048)
        # -> dropout -> (30, 10, 2048)
        # -> self.w_2 -> (30, 10, 512)是输出的shape


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and "
    "feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        # size=d_model=512
        # self_attn = MultiHeadAttention对象, first sublayer
        # feed_forward = PositionwiseFeedForward对象，second sublayer
        # dropout = 0.1 (e.g.)
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # 使用深度克隆方法，完整地复制出来两个SublayerConnection
        self.size = size  # 512

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # x shape = (30, 10, 512)
        # mask 是(batch.size, 10,10)的矩阵，类似于当前一个词w，有哪些词是w可见的
        # 源语言的序列的话，所有其他词都可见，除了"<blank>"这样的填充；
        # 目标语言的序列的话，所有w的左边的词，都可见。
        x = self.sublayer[0](x,
                             lambda x: self.self_attn(x, x, x, mask))
        # x (30, 10, 512) -> self_attn (MultiHeadAttention)
        # shape is same (30, 10, 512) -> SublayerConnection
        # -> (30, 10, 512)
        return self.sublayer[1](x, self.feed_forward)
        # x 和feed_forward对象一起，给第二个SublayerConnection


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        # 为什么还要进行一次norm 【?】
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size  # 512
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory  # (batch.size, sequence.len, 512)
        # 来自源语言序列的Encoder之后的输出，作为memory
        # 供目标语言的序列检索匹配：（类似于alignment in SMT)
        x = self.sublayer[0](x,
                             lambda x: self.self_attn(x, x, x, tgt_mask))
        # 通过一个匿名函数，来实现目标序列的自注意力编码
        # 结果扔给sublayer[0]:SublayerConnection
        x = self.sublayer[1](x,
                             lambda x: self.src_attn(x, m, m, src_mask))
        # 通过第二个匿名函数，来实现目标序列和源序列的注意力计算
        # 结果扔给sublayer[1]:SublayerConnection
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        # layer = DecoderLayer object
        # N = 6
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        # 深度copy六次DecoderLayer
        self.norm = LayerNorm(layer.size)
        # 初始化一个LayerNorm

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
            # 执行六次DecoderLayer
        return self.norm(x)
        # 执行一次LayerNorm


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        # d_model=512
        # vocab = 目标语言词表大小
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        # 定义一个全连接层，可训练参数个数是(512 * trg_vocab_size) +
        # trg_vocab_size

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
        # x 类似于 (batch.size, sequence.length, 512)
        # -> proj 全连接层 (30, 10, trg_vocab_size) = logits
        # 对最后一个维度执行log_soft_max
        # 得到(30, 10, trg_vocab_size)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    Base for this and many other models.
    """
    def __init__(self, encoder, decoder,
      src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        # Encoder对象
        self.decoder = decoder
        # Decoder对象
        self.src_embed = src_embed
        # 源语言序列的编码，包括词嵌入和位置编码
        self.tgt_embed = tgt_embed
        # 目标语言序列的编码，包括词嵌入和位置编码
        self.generator = generator
        # 生成器

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
        # 先对源语言序列进行编码，
        # 结果作为memory传递给目标语言的编码器

    def encode(self, src, src_mask):
        # src = (batch.size, seq.length)
        # src_mask 负责对src加掩码
        return self.encoder(self.src_embed(src), src_mask)
        # 对源语言序列进行编码，得到的结果为
        # (batch.size, seq.length, 512)的tensor

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt),
          memory, src_mask, tgt_mask)
        # 对目标语言序列进行编码，得到的结果为
        # (batch.size, seq.length, 512)的tensor


def subsequent_mask(size):
    "Mask out subsequent positions."
    # e.g., size=10
    attn_shape = (1, size, size)  # (1, 10, 10)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # triu: 负责生成一个三角矩阵，k-th对角线以下都是设置为0
    # 上三角中元素为1.

    return torch.from_numpy(subsequent_mask) == 0
    # 反转上面的triu得到的上三角矩阵，修改为下三角矩阵
    # [[[1, 0, 0],
    #   [1, 1, 0],
    #   [1, 1, 1]]]


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)

    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                         c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  # 使用xavier_uniform(p)来初始化二维以上的参数
    return model  # EncoderDecoder 对象


if __name__ == "__main__":
    tmp_model = make_model(30000, 30000, 6)
    # src_vocab_size=30000, tgt_vocab_size=30000, N=6
    # None
    for name, param in tmp_model.named_parameters():
        if param.requires_grad:
            print (name, param.data.shape)
        else:
            print ('no gradient necessary', name, param.data.shape)
