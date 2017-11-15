import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class Encoder(nn.Module):
    """
    encoder of NMT
    """

    def __init__(self, src_vocab_size, pad_idx, src_embedding_size=300, hidden_size=512):
        """
        Args:
            src_vocab_size(int): source language vocabulary size
            src_embedding_size(int): word embedding size for source language
            hidden_size(int): encoder's hidden size, should be equal to decoder's hidden size // 2
        """
        super(Encoder, self).__init__()
        self.embeddings = nn.Embedding(src_vocab_size, src_embedding_size, padding_idx=pad_idx)
        self.rnn = nn.LSTM(src_embedding_size, hidden_size, bidirectional=True)

    def forward(self, input, mask = None):
        """
        Args:
            input(LongTensor): a batch of source language sentences - shape: (src_len, batch)
            mask(ByteTensor): a batch of source language sentences' mask - shape: (src_len, batch)

        Returns:
            output(FloatTensor): output of encoder's RNN - shape: (src_len, batch, 2 * hidden_size)
            hidden(tuple): (h_n, c_n)
                    h_n(FloatTensor): final hidden state - shape: (1, batch, 2 * hidden_size)
                    c_n(FloatTensor): final LSTM cell state - shape: (1, batch, 2 * hidden_size)
        """
        emb = self.embeddings(input)
        
        packed = emb
        if mask is not None:
            lengths = torch.sum(mask, dim=0).data.tolist()
            packed = pack(emb, lengths)
        output, hidden = self.rnn(packed, None)
        if mask is not None:
            output, output_lengths = unpack(output)

        hidden = tuple([torch.cat([h[0::2], h[1::2]], 2) for h in hidden])
        return output, hidden


class Decoder(nn.Module):
    """
    decoder of NMT
    hidden_size = encoder's hidden size // 2
    """

    def __init__(self, trg_vocab_size, pad_idx, trg_embedding_size=300, hidden_size=1024):
        """
        Args:
            trg_vocab_size(int): target language vocabulary size
            trg_embedding_size(int): word embedding size for target language
            hidden_size(int): decoder's hidden size

        """
        super(Decoder, self).__init__()
        self.embeddings = nn.Embedding(trg_vocab_size, trg_embedding_size, padding_idx=pad_idx)
        self.rnn = nn.LSTM(trg_embedding_size + hidden_size, hidden_size)
        self.attn = Attention(hidden_size)

    def forward(self, input, encoder_output, last_output, last_hidden):
        """
        Args:
            input(LongTensor): a batch of target language at time  t - shape: (1, batch)
            encoder_output(FloatTensor): output of encoder's RNN - shape: (src_len, batch, hidden_size)
            last_output(FloatTensor): output of decoder's attention layer at time t - 1 - shape: (1, batch, hidden)
            last_hidden(tuple): hidden state of decoder's RNN at time t - 1

        Returns:
            outputs(FloatTensor): decoder's attention layer's output - shape: (1, batch, hidden_size)
        """
        emb = self.embeddings(input)

        # concat with last rnn output - shape: (1, batch, hidden_size + trg_embedding_size)
        rnn_input = torch.cat([emb, last_output], 2)

        # rnn_output - shape: (1, batch, hidden_size)
        rnn_output, hidden = self.rnn(rnn_input, last_hidden)
        output = self.attn(rnn_output, encoder_output)

        return output, hidden


class Attention(nn.Module):
    """
    Luong Attention
    """

    def __init__(self, size):
        """
        Args:
            size: hidden size of decoder
        """
        super(Attention, self).__init__()
        self.linear_in = nn.Linear(size, size, bias=False)
        self.linear_out = nn.Linear(size * 2, size, bias=False)
        self.norm = nn.Softmax(dim=2)

    def forward(self, input, encoder_output):
        """
        run this one step at a time
        Args:
            input(FloatTensor): output of decoder's RNN at time step t - shape: (1, batch, hidden_size)
            encoder_output(FloatTensor): encoder's output - shape: (src_len, batch, hidden_size)
        
        Returns:
            c(FloatTensor): context vector c_t - shape: (1, batch, hidden_size)
        """
        # -> (batch, 1, hidden_size)
        input = input.transpose(0, 1)
        # -> (batch, src_len, hidden_size)
        encoder_output = encoder_output.transpose(0, 1)

        input_ = self.linear_in(input)
        # attention weight for each source word - shape: (batch, 1, src_len)
        attn_weight = torch.bmm(input_, encoder_output.transpose(1, 2))
        # normalize on src_len
        attn_weight = self.norm(attn_weight)

        # context vector c_t weighted average over all the source hidden states - shape: (batch, 1, hidden_size)
        c = torch.bmm(attn_weight, encoder_output)
        # concat with input - shape: (batch, 1, 2 * hidden_size)
        c = torch.cat((c, input), dim=2)
        # linear out - shape: (batch, 1, hidden_size)
        c = torch.tanh(self.linear_out(c))

        return c.transpose(0, 1)


class NMT(nn.Module):
    def __init__(self, src, trg, src_emb=300, trg_emb=300, hidden_size=1024):
        """
        Args:
            src_vocab_size(int):
            trg_vocab_size(int):a
        """
        super(NMT, self).__init__()
        self.encoder = Encoder(len(src), src.stoi["<blank>"], src_emb, hidden_size / 2)
        self.decoder = Decoder(len(trg), trg.stoi["<blank>"], trg_emb, hidden_size)
        self.out = nn.Linear(hidden_size, len(trg))
        self.norm = nn.LogSoftmax(dim=2)
        self.SOS = Variable(torch.LongTensor([trg.stoi["<s>"]]), volatile=True, requires_grad=False)
        self.EOS = Variable(torch.LongTensor([trg.stoi["</s>"]]), volatile=True, requires_grad=False)
        self.MAX_LENGTH = 50
    def forward(self, src, src_mask, trg, trg_len):
        """
        Args:
            src(LongTensor): a batch of source language sentences - shape: (src_len, batch)
            trg(LongTensor): a batch of target language sentences - shape: (trg_len, batch)
        
        Returns:
            output(FloatTensor): normalized word distribution - shape: (trg_len, batch, trg_vocab_size)
        """
        
        encoder_output, hidden = self.encoder(src, src_mask)
        output = []
        decoder_output = Variable(torch.zeros(1, encoder_output.size(1), encoder_output.size(2)),
                                  requires_grad=False)
        if src.is_cuda:
            decoder_output = decoder_output.cuda()
            self.SOS = self.SOS.cuda()
            self.EOS = self.EOS.cuda()    
            
        if trg is not None:
            trg = trg[:-1]#exclude EOS
            for t in trg.split(1):
                input = t
                decoder_output, hidden = self.decoder(input, encoder_output, decoder_output, hidden)
                output.append(decoder_output)
        else:
            self.SOS = self.SOS.expand(1, src.size(1))
            self.EOS = self.EOS.expand(1, src.size(1))
            
            MAX_LENGTH = self.MAX_LENGTH if trg_len is None else trg_len
            input = self.SOS
            for i in range(MAX_LENGTH):
                decoder_output, hidden = self.decoder(input, encoder_output, decoder_output, hidden)
                output.append(decoder_output)
                input = torch.max(self.norm(self.out(decoder_output)), dim = 2)[1]
                if trg_len is None and input.equal(self.EOS):
                    break
        output = torch.stack(output).squeeze(1)
        return self.norm(self.out(output))