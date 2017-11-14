from __future__ import division
from word_alignment import *

class diagonal_align(Model):
    def __init__(self, bitext):
        """
        diagonal_tension = lambda
        n_target = sum(len(e))
        size_counts is dict where key is sentences length and value is the count used for lambda update
        p0 is the prob of null
        :return:
        """
        Model.__init__(self, bitext)

        self.size_counts = defaultdict(int)
        self.n_target = 0
        for (f, e) in bitext:
            self.size_counts[(len(f), len(e))] += 1
            self.n_target += len(e)
        self.diagonal_tension = 4
        self.p0 = 0.08

    def training(self):
        opts = {"display":5000,"MAX_ITERATION": 7}
        for iter in range(opts["MAX_ITERATION"]):
            #emp is a fixed num for each iteration
            emp = likelihood = 0
            sys.stderr.write("{:d} iteration(s)".format(iter + 1))
            # EM
            count = dict.fromkeys(self.fe_dict, 0.0)
            total = dict.fromkeys(self.french_dict, 0.0)
            for (n, (f_sent, e_sent)) in enumerate(self.bitext):
                if n % opts["display"] == 0:
                    sys.stderr.write(".".format(n))

                probs = [0] * len(f_sent)
                #target sentence
                for (j, e_j) in enumerate(e_sent):
                    # null token
                    probs[0] = self.t[(None, e_j)] * self.p0

                    #normalization factor
                    az = self.Z(j + 1, len(f_sent) - 1, len(e_sent)) / (1 - self.p0)
                    for (i, f_i) in enumerate(f_sent):
                        if f_i == None:
                            continue
                        prob = self.p(i + 1, j + 1, len(f_sent) - 1, len(e_sent)) / az
                        probs[i + 1] = self.t[(f_i, e_j)] * prob

                    s = sum(probs)
                    likelihood -= math.log(s)
                    # fractional count
                    p = probs[0] / s
                    count[(None, e_j)] += p
                    total[None] += p
                    for (i, f_i) in enumerate(f_sent):
                        if f_i == None:
                            continue
                        p = probs[i + 1] / s
                        count[(f_i, e_j)] += p
                        total[f_i] += p
                        emp += self.feature(i + 1, j, len(f_sent) - 1, len(e_sent)) * p
            sys.stderr.write("\n")
            for (f, e) in self.fe_dict:
                self.t[(f, e)] = count[(f, e)] / total[f]
            emp /= self.n_target
            #update lambda
            if iter > 0 and iter < opts["MAX_ITERATION"] - 1:
                for x in range(8):
                    sys.stderr.write("diagonal tension: {:0.2f} optimized: {:d}\n".format(self.diagonal_tension,x + 1))
                    delta = 0
                    for k in self.size_counts:
                        for j in range(k[1]):
                            delta += self.size_counts[k] * self.compute_d_logz(j + 1, k[0] - 1, k[1])

                    self.diagonal_tension += (emp - delta / self.n_target) * 20
                    if self.diagonal_tension < 0.1:
                        self.diagonal_tension = 0.1
                    if self.diagonal_tension > 20:
                        self.diagonal_tension = 20

            sys.stderr.write("cross entropy: {:.2f}\n".format(likelihood))

    def geo_series(self, g, a, r, d, m):
        """
        detail in compute_d_logz
        :param g: prob
        :param a: feature
        :param r:
        :param d:
        :param m:
        :return:
        """
        g_ = g * (r ** m)
        a_ = d * (m - 1) + a
        rm = r - 1
        return (a_ * g_ - a * g) / rm - d * (g_ - g * r) / (rm * rm)

    def compute_d_logz(self, j, m, n):
        """
        compute the gradient sub of z using for z's update
        :param j: index of target word
        :param m: length of source
        :param n: length of target
        :return:
        """
        floor = j * m // n
        ceil = floor + 1
        r = math.exp(- self.diagonal_tension / m)
        d = -1 / m
        num_top = m - floor
        top = btm = 0
        if num_top != 0:
            top = self.geo_series(self.p(ceil, j, m, n), self.feature(ceil, j, m, n), r, d, num_top)
        if floor != 0:
            btm = self.geo_series(self.p(floor, j, m, n), self.feature(floor, j, m, n), r, d, floor)
        return (top + btm) / self.Z(j, n, m)

    def p(self, i, j, m, n):
        """
        exp(lambda * feature)
        :param i: index of target word
        :param j: index of source word
        :param m: length of source
        :param n: length of target
        :return:
        """
        return math.exp(self.diagonal_tension * self.feature(i, j, m, n))

    def feature(self, i, j, m, n):
        """
        :param i: index of source word
        :param j: index of target word
        :param m: length of source
        :param n: length of target
        :return: -|i/m-j/n|
        """
        return -abs(i / m - j / n)

    def Z(self, j, m, n):
        """
        normalization factor
        :param j: index of target word
        :param m: length of source
        :param n: length of target
        :return:
        """
        floor = j * m // n
        r = math.exp(-self.diagonal_tension / m)
        btm = top = 0
        if floor != 0:
            btm = self.p(floor, j, m, n) * (1 - r ** floor)
        if m - floor != 0:
            top = self.p(floor + 1, j, m, n) * (1 - r ** (m - floor))
        return (top + btm) / (1 - r)


if __name__ == "__main__":
    f_data, e_data, n = open_file()
    bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:n]]
    # # bitext = [[["das", "haus"], ["the", "house"]], [["das", "buch"], ["the", "book"]], [["ein", "buch"], ["a", "book"]]]
    model = diagonal_align(bitext)
    model.training()
    model.get_alignment()