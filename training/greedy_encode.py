import torch
from torch.autograd import Variable

from Net.net import subsequent_mask


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    # 源语言的一个batch
    # 执行encode编码工作，得到memory
    # shape=(batch.size, src.seq.len, d_model)

    # src = (1,4), batch.size=1, seq.len=4
    # src_mask = (1,1,4) with all ones
    # start_symbol=1

    print('memory={}, memory.shape={}'.format(memory,
                                              memory.shape))
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 最初ys=[[1]], size=(1,1); 这里start_symbol=1
    print('ys={}, ys.shape={}'.format(ys, ys.shape))
    for i in range(max_len - 1):  # max_len = 5
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        # memory, (1, 4, 8), 1=batch.size, 4=src.seq.len, 8=d_model
        # src_mask = (1,1,4) with all ones
        # out, (1, 1, 8), 1=batch.size, 1=seq.len, 8=d_model
        print('out={}, out.shape={}'.format(out, out.shape))
        prob = model.generator(out[:, -1])
        # pick the right-most word
        # (1=batch.size,8) -> generator -> prob=(1,5) 5=tgt.vocab.size
        # -1 for ? only look at the final (out) word's vector
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        # word id of "next_word"
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)],
                       dim=1)
        # ys is in shape of (1,2) now, i.e., 2 words in current seq
    return ys


# if True:
#     model.eval()
#     src = Variable(torch.LongTensor([[1, 2, 3, 4]]))
#     src_mask = Variable(torch.ones(1, 1, 4))
#     print(greedy_decode(model, src, src_mask, max_len=5,
#                         start_symbol=1))
