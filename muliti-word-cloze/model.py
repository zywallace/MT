import torch, math
import torch.nn as nn
from torch.autograd import Variable


class Linear(nn.Module):
    """
    linear transformation layer with learnable variable: weight and bias
    """

    def __init__(self, in_size, out_size):
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.zeros(in_size, out_size))
        self.b = nn.Parameter(torch.zeros(out_size))
        # reset uniformly
        stdv = 1. / math.sqrt(self.w.size(0))
        self.w.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return torch.matmul(input, self.w) + self.b


class Embedding(nn.Module):
    """
    word embedding layer with learnable variable: C
    """

    def __init__(self, in_size, out_size):
        super(Embedding, self).__init__()
        self.C = nn.Parameter(torch.zeros(in_size, out_size))
        # reset uniformly
        stdv = 1. / math.sqrt(self.C.size(0))
        self.C.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return self.C[input.data, :]


class LogSoftmax(nn.Module):
    """
    log softmax layer with no learnable variable
    """

    def forward(self, input):
        X = torch.exp(input - torch.max(input, dim=2, keepdim=True)[0])
        return torch.log(X) - torch.log(torch.sum(X, dim=2, keepdim=True))


class RNN(nn.Module):
    """
    RNN layer
    h_t = f(w_h*h_{t-1} + w_i*x_t + b)
    """

    def __init__(self, in_size, out_size, bi_directional=False):
        super(RNN, self).__init__()
        self.hidden_size = out_size
        self.i2h = Linear(in_size, out_size)
        self.h2h = Linear(out_size, out_size)
        # self.activation = torch.sigmoid
        self.activation = torch.tanh
        self.bi_directional = bi_directional
        if bi_directional:
            self.i2h_back = Linear(in_size, out_size)
            self.h2h_back = Linear(out_size, out_size)

    def forward(self, input):
        T = input.data.shape[0]

        h = [Variable(torch.zeros(input.data.shape[1], self.hidden_size))]
        if input.is_cuda:
            h[0] = h[0].cuda()
        # for each time step
        for t in range(T):
            h.append(self.activation(self.i2h(input[t]) + self.h2h(h[-1])))

        if self.bi_directional:
            h_back = [Variable(torch.zeros(input.data.shape[1], self.hidden_size))]
            if input.is_cuda:
                h_back[0] = h_back[0].cuda()
            # from right to the left
            for t in range(T - 1, -1, -1):
                h_back.append(self.activation(self.i2h_back(input[t]) + self.h2h_back(h_back[-1])))
            # reverse so h_back[-1] is the init state
            h_back = h_back[::-1]
            h = torch.stack(h[:-1], 0)  # shift ignore the last token (end of sent)
            h_back = torch.stack(h_back[1:], 0)  # ignore the first token(start of sent)
            return torch.cat((h, h_back), dim=2)
        else:
            return torch.stack(h[1:], 0)


class RNNLM(nn.Module):
    def __init__(self, vocab_size, bi_directional=False):
        super(RNNLM, self).__init__()
        self.input_size = vocab_size
        self.embedding_size = 128
        num_dir = 2 if bi_directional else 1
        self.hidden_size = 64

        self.layers = nn.ModuleList()
        self.layers.append(Embedding(self.input_size, self.embedding_size))
        self.layers.append(Dropout(.2))
        self.layers.append(RNN(self.embedding_size, (int)(self.hidden_size / num_dir), bi_directional=bi_directional))
        self.layers.append(Dropout())
        self.layers.append(Linear(self.hidden_size, self.input_size))
        self.layers.append(LogSoftmax())

    def forward(self, input_batch):
        """
        just treat it like a sequential feed-forward NN
        input shape seq_len, batch_size
        ouput shape sequence_length, batch_size, vocab_size
        """
        output = input_batch
        for layer in self.layers:
            output = layer(output)
        return output

    def train(self, mode=True):
        """

        :param mode: True = training_mode otherwise evaluation_mode
        :return:
        """
        self.training = mode
        for layer in self.layers:
            layer.train(mode)

    def eval(self):
        self.train(False)


class Dropout(nn.Module):
    """
    A inverted dropout layer
    """

    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout prob has to be between 0 and 1, but got {}".format(p))
        self.p = p  # self.p is the drop rate, if self.p is 0, then it's a identity layer

    def forward(self, input):
        if self.training:
            p = self.p
            if p == 1:
                # all units would be dropped
                mask = Variable(torch.zeros(input.size()).fill_(0))
            else:
                # each unit would be dropped with prob of p and scale by p
                mask = Variable(torch.zeros(input.size()).bernoulli_(1 - p).div_(1 - p))
            if input.is_cuda:
                mask = mask.cuda()
            return input * mask
        else:
            return input
