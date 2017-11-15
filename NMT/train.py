from __future__ import division
from data_ops import load_data, next_batch, flat
import opts
import dill
import argparse, logging
import torch
from torch import cuda
from model import NMT

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)


def iter(src, trg, model, criterion, optimizer, use_cuda, options, training):
    num_batch = len(src[0])
    loss = 0
    i = 0
    for src_batch, src_mask, trg_batch, trg_mask in next_batch(src, trg, use_cuda, num_batch, training):
        # trg_len, batch, trg_vocab_size
        out_batch = model(src_batch, src_mask, trg_batch)

        # trg_batch - shape: (N) out_batch - shape: (N, trg_vocab_size)
        # exclude SOS
        trg_batch, out_batch = flat(out_batch, trg_batch[1:,:], trg_mask[1:,:])

        batch_loss = criterion(out_batch, trg_batch)

        if training:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        loss += batch_loss.data[0]
        i += 1
        if training and i % options.show == 0:
            logging.info("Average loss value per instance is {:.5f} at batch {}".format(loss / i, i))
    loss /= num_batch
    return loss


def main(options):
    use_cuda = (len(options.gpuid) >= 1)
    logging.info("loading data")
    src, trg = load_data(options)
    logging.info("loading finished\n" + src.__str__() + "\n" + trg.__str__())


    nmt = NMT(src.vocab, trg.vocab)

    if use_cuda:
        logging.info("init NMT with cuda enable")
        nmt.cuda()
        cuda.set_device(options.gpuid[0])
    else:
        logging.info("init NMT with cuda disable")

    criterion = torch.nn.NLLLoss()
    optimizer = eval("torch.optim." + options.optimizer)(nmt.parameters(), options.learning_rate)

    prev_loss = float("inf")
    for epoch_i in range(options.epochs):
        logging.info("At {0}-th epoch.".format(epoch_i))
        # train
        iter(src.data["train"], trg.data["train"], nmt, criterion, optimizer, use_cuda, options, training=True)

        loss = iter(src.data["dev"], trg.data["dev"], nmt, criterion, optimizer, use_cuda, options, training=False)
        logging.info("Average loss value per instance is {0} at the end of epoch {1}".format(loss, epoch_i))
        torch.save(nmt.state_dict(), open(options.model_file + ".nll_{0:.2f}.epoch_{1}".format(loss, epoch_i), 'wb'),
                   pickle_module=dill)
        logging.info("model saved at " + options.model_file + ".nll_{0:.2f}.epoch_{1}".format(loss, epoch_i))
        if abs(prev_loss - loss) < options.estop:
            logging.info(
                "Early stopping triggered with previous: {0:.4f} and current: {1:.4f})".format(prev_loss, loss))
            break
        prev_loss = loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Starter code for JHU CS468 Machine Translation HW5.")
    opts.nmt_opts(parser)
    opt = parser.parse_args()
    logging.info(opt.__str__())
    main(opt)
