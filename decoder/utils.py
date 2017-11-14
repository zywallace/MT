#!/usr/bin/env python
# -*- coding: utf-8 -*-
import optparse, sys
from collections import namedtuple

phrase = namedtuple("phrase", "english, logprob")
ngram_stats = namedtuple("ngram_stats", "logprob, backoff")
hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, coverage, source, index")


def opt_parse():
    """
    :return: object of parsed arguments
    """
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--input", dest="input", default="data/input",
                         help="File containing sentences to translate (default=data/input)")
    optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm",
                         help="File containing translation model (default=data/tm)")
    optparser.add_option("-l", "--language-model", dest="lm", default="data/lm",
                         help="File containing ARPA-format language model (default=data/lm)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=1, type="int",
                         help="Number of sentences to decode (default=no limit)")
    optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1000, type="int",
                         help="Limit on number of translations to consider per phrase (default=1)")
    optparser.add_option("-s", "--stack-size", dest="s", default=50000, type="int", help="Maximum stack size (default=1)")
    optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,
                         help="Verbose mode (default=off)")
    return optparser.parse_args()[0]


def TM(filename, k):
    sys.stderr.write("Reading translation model from %s...\n" % (filename,))
    tm = {}
    for line in open(filename).readlines():
        (f, e, logprob) = line.strip().split(" ||| ")
        tm.setdefault(tuple(f.split()), []).append(phrase(e, float(logprob)))
    for f in tm:  # prune all but top k translations
        tm[f].sort(key=lambda x: -x.logprob)
        del tm[f][k:]
    return tm


def next_phrase(f, v, tm):
    """
    generator of available phrases given currently translation status v
    :param f: input sentence
    :param v: boolean list of coverage v[i] == True means f[i] has been translated
    :param tm: translation model
    :return: a phrase, its start index and its size
    """
    for i in range(len(f)):
        for j in range(i + 1, len(f) + 1):
            if not any(v[i:j]) and f[i:j] in tm:
                for phrase in tm[f[i:j]]:
                    yield phrase, i, j - i


class LM:
    def __init__(self, filename):
        sys.stderr.write("Reading language model from %s...\n" % (filename,))
        self.table = {}
        for line in open(filename):
            entry = line.strip().split("\t")
            if len(entry) > 1 and entry[0] != "ngram":
                (logprob, ngram, backoff) = (
                    float(entry[0]), tuple(entry[1].split()), float(entry[2] if len(entry) == 3 else 0.0))
                self.table[ngram] = ngram_stats(logprob, backoff)

    def begin(self):
        return ("<s>",)

    def score(self, state, word):
        ngram = state + (word,)
        score = 0.0
        while len(ngram) > 0:
            if ngram in self.table:
                return (ngram[-2:], score + self.table[ngram].logprob)
            else:  # backoff
                score += self.table[ngram[:-1]].backoff if len(ngram) > 1 else 0.0
                ngram = ngram[1:]
        return ((), score + self.table[("<unk>",)].logprob)

    def end(self, state):
        return self.score(state, "</s>")[1]

    def init_stacks(self, f):
        initial_hypothesis = hypothesis(0.0, self.begin(), None, None, [False] * len(f), None, 0)
        stacks = [{} for _ in f] + [{}]
        stacks[0][self.begin()] = initial_hypothesis
        return stacks


def get_data():
    """
    :return: argument, translation model, language model and french input
    """
    opts = opt_parse()
    tm = TM(opts.tm, opts.k)
    lm = LM(opts.lm)
    french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]
    # french = [tuple(line.strip().split()) for line in open("data/test").readlines()[:opts.num_sents]]

    for word in set(sum(french, ())):
        if (word,) not in tm:
            tm[(word,)] = [phrase(word, 0.0)]

    return opts, lm, tm, french


def hypothesis2list(h):
    """
    used only for greedy decoder, some operations like swap and merge would be painful using
    hypothesis structure
    :param h: hypothesis
    :return: a list of translation tuple
    eg. [(phrase, tuple of source segment),]
    """
    li = []
    ptr = h
    while ptr.predecessor:
        li.append((ptr.source, ptr.phrase, ptr.index))
        ptr = ptr.predecessor
    li.reverse()
    return li

if __name__ == "__main__":
    data = get_data()
