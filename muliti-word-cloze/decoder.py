import dill
import torch
from torch.autograd import Variable


def cloze_decode(test_data=None, vocab=None, rnnlm=None):
    if rnnlm == None:
        # load from saved model
        rnnlm = torch.load(open("gpu.nll_3.99.epoch_5", 'rb'), pickle_module=dill)

    if test_data == None:
        _, _, test_data, vocab = torch.load(open("data/hw4_data.bin", 'rb'), pickle_module=dill)

    rnnlm.eval()
    use_cuda = next(rnnlm.parameters()).is_cuda
    for sent in test_data:
        if use_cuda:
            sent = sent.cuda()
        input = Variable(torch.unsqueeze(sent, 1))
        _, output = torch.max(torch.squeeze(rnnlm(input)), 1)  # torch.max()[1] is equivalent to argmax

        indices = output[sent == 1].data  # 1 is the index of <blank> in vocab
        result = [vocab.itos[index].encode("utf-8") for index in indices]
        print(' '.join(result))
    rnnlm.train()


if __name__ == "__main__":
    cloze_decode()
