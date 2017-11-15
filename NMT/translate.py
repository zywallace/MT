from sys import stderr
from opts import translation_opts
import argparse
from model import NMT
import torch
from torch.autograd import Variable


def main(options):
    _, _, src_test, src_vocab = torch.load(open(options.data_file + "." + options.src_lang, 'rb'))
    _, _, trg_test, trg_vocab = torch.load(open(options.data_file + "." + options.trg_lang, 'rb'))

    nmt = NMT(src_vocab, trg_vocab, opt=options)
    param = torch.load(options.model_file, lambda storage, loc: storage)
    nmt.load_state_dict(param)
    for src in src_test:
        output = nmt(Variable(src.unsqueeze(1), volatile=True), None, None, None).squeeze()
        trg_index = torch.max(output, dim=1)[1]
        trg = []
        trg_index = trg_index.data.tolist()
        for index in trg_index:
            trg.append(trg_vocab.itos[index])
        print(" ".join(trg))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    translation_opts(parser)
    opt = parser.parse_args()
    stderr.write(opt.__str__())
    main(opt)
