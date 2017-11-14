from __future__ import print_function
from utils import *
from itertools import chain


class decoder:
    def __init__(self):
        self.opts, self.lm, self.tm, self.input = get_data()
        self.seed = None

    def decode(self):
        sys.stderr.write("Decoding...\n")
        for index, f in enumerate(self.input, start=1):
            sys.stderr.write("{:d}: length: {:d}\n".format(index, len(f)))
            self.beam_search_decoder(f)
            r = self.greedy_decoder(f)
            print(" ".join([e.english for f, e, i in r]))

    def greedy_decoder(self, source):
        """
        greedy decoder as described in
        http://www.iro.umontreal.ca/~felipe/bib2webV0.81/cv/papers/paper-tmi-2007.pdf
        :return: a maximized candidate translation
        """
        cur = self.seed if self.seed else self.beam_search_decoder(source)
        MAX_ITER = int(1e3)
        sys.stderr.write("seed score:{:2f}\n".format(self.score(cur, source)))
        for iter in range(1, MAX_ITER + 1):
            if iter % 100 == 0:
                sys.stderr.write("{:05d} iterations finished".format(iter))
            s_cur = self.score(cur, source)
            s = s_cur
            for h in self.neighbor(cur):
                c = self.score(h, source)
                if c > s:
                    sys.stderr.write("new best score:{:2f}\n".format(c))
                    s = c
                    best = h
            if s == s_cur:
                return cur
            else:
                cur = best
        return cur

    def neighbor(self, cur):
        """
        set of operations to transform current translation to possible better ones
        there are six of them in original paper and we've implemented all of them
        which are swap, sleep, replace, split, bi-replace and move
        :param cur:
        :return:
        """
        return chain(self.move(cur), self.swap(cur), self.replace(cur), self.bi_replace(cur), self.split(cur),
                     self.merge(cur))

    def move(self, cur):
        """
        whenever two adjacent source phrases are translated by phrases that are distant (more than 3)
        we consider moving one of the translation closer to the other
        for simplicity, we just swap the translation of left one, let's say, target[i], with target[i + 1]
        :param cur:
        :return:
        """
        index = [i for f, e, i in cur]
        # index[2, 1] should be come [1, 0]
        index = sorted(range(len(index)), key=lambda i: index[i])[:-1]
        for i in range(len(index) - 1):
            if abs(index[i] - index[i + 1]) > 3:
                r = cur[:]
                x = min(index[i], index[i + 1])
                r[x], r[x + 1] = r[x + 1], r[x]
                yield r

    def swap(self, cur):
        """
        swap two adjacent phrases.
        :param cur:
        :return:
        """
        for i in range(len(cur) - 1):
            r = cur[:]
            r[i], r[i + 1] = r[i + 1], r[i]
            yield r

    def replace(self, cur):
        """
        change the translation given for a specific source segment
        :param cur:
        :return:
        """
        for i, p in enumerate(cur):
            f, e = p[0], p[1]
            for phrase in self.tm[f]:
                if phrase != e:
                    r = cur[:]
                    r[i] = (f, phrase, p[2])
                    yield r

    def bi_replace(self, cur):
        """
        similar to replace, change two adjacent source phrases's translation
        must change both of their translations at the same time, otherwise it's just replace()
        :param cur:
        :return:
        """
        for i in range(len(cur) - 1):
            f1, e1, i1 = cur[i]
            f2, e2, i2 = cur[i + 1]
            for p1 in self.tm[f1]:
                for p2 in self.tm[f2]:
                    if p1 != e1 and p2 != e2:
                        r = cur[:]
                        r[i] = (f1, p1, i1)
                        r[i + 1] = (f2, p2, i2)
                        yield r

    def split(self, cur):
        """
        split source segment with more than one words
        :param cur:
        :return:
        """
        for i, p in enumerate(cur):
            f, e = p[0], p[1]
            for j in range(1, len(f)):
                f1, f2 = f[:j], f[j:]
                if f1 in self.tm and f2 in self.tm:
                    for p1 in self.tm[f1]:
                        for p2 in self.tm[f2]:
                            r = cur[:]
                            r[i] = (f1, p1, p[2])
                            r.insert(i + 1, (f2, p2, p[2] + j))
                            yield r

    def merge(self, cur, k=2):
        """
        merge k adjacent segements
        default k = 2
        :param cur:
        :return:
        """
        assert k > 1
        for i in range(len(cur) - k + 1):
            t = ()
            for j in range(k):
                t += cur[i + j][0]
            if t in self.tm:
                for p in self.tm[t]:
                    r = cur[:]
                    r[i] = (t, p, cur[i][2])
                    del r[i + 1:i + k]
                    yield r

    def score(self, translation, source):
        """
        score the current translation, it should be similar to the log_prob method
        :param translation:
        :param source:
        :return:
        """
        prob = size = 0
        lm_state = self.lm.begin()
        for f, e, _ in translation:
            size += len(f)
            prob, lm_state = self.log_prob(prob, lm_state, source, e, size)
        return prob

    def log_prob(self, prob, state, source, phrase, size):
        """
        calculate the log prob (translation + language) of the current hypothesis
        :param prob: current hypothesis's log prob
        :param state: current h's lm state
        :param source: input sentence
        :param phrase: translation model
        :param size: number of translated words
        :return: tm_prob + lm_prob
        """
        logprob = prob + phrase.logprob
        lm_state = state
        for word in phrase.english.split():
            (lm_state, word_logprob) = self.lm.score(lm_state, word)
            logprob += word_logprob
        logprob += self.lm.end(lm_state) if size == len(source) else 0.0
        return logprob, lm_state

    def beam_search_decoder(self, source):
        """
        beam search decoder, support global reordering
        :param source: input sentence
        :return: None
        """
        sys.stderr.write(" ".join(source) + "\n")
        stacks = self.lm.init_stacks(source)
        for i, stack in enumerate(stacks[:-1]):
            for h in sorted(stack.itervalues(), key=lambda h: -h.logprob)[:self.opts.s]:
                v = h.coverage
                for phrase, index, size in next_phrase(source, v, self.tm):
                    logprob, lm_state = self.log_prob(h.logprob, h.lm_state, source, phrase, size + i)

                    v[index:index + size] = [True] * size
                    new_hypothesis = hypothesis(logprob, lm_state, h, phrase, v[:], source[index:index + size], index)
                    v[index:index + size] = [False] * size

                    key = lm_state
                    # key = tuple(new_hypothesis.coverage)
                    if key not in stacks[i + size] or stacks[i + size][
                        key].logprob < logprob:
                        stacks[i + size][key] = new_hypothesis
            sys.stderr.write("covered {:d} words\n".format(i + 1))

        self.seed = hypothesis2list(max(stacks[-1].itervalues(), key=lambda h: h.logprob))
        return self.seed


if __name__ == "__main__":
    decoder = decoder()
    decoder.decode()
