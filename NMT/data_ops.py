import torch
from utils.tensor import advanced_batchize, advanced_batchize_no_sort
from utils.rand import srange
from torch.autograd import Variable


class Lang:
    def __init__(self, test, vocab, batch_size):
        self.vocab = vocab
        self.data = {"test": test}
        self.batch_size = batch_size

    def __str__(self):
        return "There are {} words, {} train sentences, {} dev sentences, {} test sentences".\
            format(len(self.vocab), len(self.data["train"][0]) * self.batch_size,
                   len(self.data["dev"][0]) * self.batch_size, len(self.data["test"]))


def load_data(options):
    src_data = torch.load(open(options.data_file + "." + options.src_lang, 'rb'))
    trg_data = torch.load(open(options.data_file + "." + options.trg_lang, 'rb'))

    size = options.batch_size
    src = Lang(src_data[2], src_data[3], size)
    trg = Lang(trg_data[2], trg_data[3], size)
    src_blank = src.vocab.stoi["<blank>"]
    trg_blank = trg.vocab.stoi["<blank>"]

    src.data["train"], trg.data["train"] = make_batches(src_data[0], trg_data[0], size, src_blank, trg_blank)
    src.data["dev"], trg.data["dev"] = make_batches(src_data[1], trg_data[1], size, src_blank, trg_blank)

    return src, trg


def make_batches(src, trg, size, src_blank, trg_blank):
    src_batch, src_mask, sort_index = advanced_batchize(src, size, src_blank)
    trg_batch, trg_mask = advanced_batchize_no_sort(trg, size, trg_blank, sort_index)
    return [src_batch, src_mask], [trg_batch, trg_mask]


def next_batch(src, trg, use_cuda, range, training):
    for i in srange(range):
        src_batch = Variable(src[0][i])  # of size (src_seq_len, batch_size)
        trg_batch = Variable(trg[0][i])  # of size (src_seq_len, batch_size)
        src_masks = Variable(src[1][i])
        trg_masks = Variable(trg[1][i])
        if not training:
            src_batch.volatile = True
            trg_batch.volatile = True
            src_masks.volatile = True
            src_masks.volatile = True

        if use_cuda:
            src_batch = src_batch.cuda()
            trg_batch = trg_batch.cuda()
            src_masks = src_masks.cuda()
            trg_masks = trg_masks.cuda()
        yield src_batch, src_masks, trg_batch, trg_masks


def flat(output, trg, trg_mask):
    size = output.size(2)
    trg_mask = trg_mask.view(-1)
    trg = trg.view(-1)
    trg = trg.masked_select(trg_mask)
    trg_mask = trg_mask.unsqueeze(1).expand(len(trg_mask), size)
    output = output.view(-1, size)
    output = output.masked_select(trg_mask).view(-1, size)
    return trg, output
