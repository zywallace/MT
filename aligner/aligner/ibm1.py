import sys
import random
from collections import defaultdict


class IBM1:
    def __init__(self, f_data, e_data, num_sents, threshold):
        """
        init training file by number of sentences used for training
        :param threshold: min difference between t(e|f) and 0|1 in order to be recognized as convergence
        """
        self.threshold = threshold
        # Strip whitespace characters from both ends of each sentence and split by whitespace
        # bitext -> [[['s1f1', 's1f2'], ['s1e1', 's1e2']], [['s2f1', 's2f2'], ['s2e1', 's2e2']], ...]
        self.bitext = [[sentence.strip().split() for sentence in pair]
                       # zip() -> [(sen1f, sen1e), (sen2f, sen2e), ...]; file object are automatically split by \n.
                       for pair in zip(open(f_data), open(e_data))[:num_sents]]
        # init dictionaries
        self.total_f = defaultdict(float)
        # self.e_count = defaultdict(float)
        self.count_ef = defaultdict(float)
        self.t_ef = defaultdict(float)  # t(e|f), translation probability

    def train(self):
        sys.stderr.write("Training with Dice's coefficient...\n")
        # Generate and count all possible translation combination
        for (n, (f, e)) in enumerate(self.bitext):  # enum() -> [(0,obj0), (1,obj1), ...]
            # f -> ['s1f1', 's1f2']
            for e_j in set(e):
                # match with every possible english word
                for f_i in set(f):
                    ef = e_j + r'|' + f_i
                    # count freq. of this match
                    self.count_ef[ef] = 1.0 if (ef not in self.count_ef) else (self.count_ef.get(ef) + 1)
                    self.total_f[f_i] = 1.0 if (f_i not in self.total_f) else (self.total_f.get(f_i) + 1)
        # init t(e|f) uniformly
        for ef in self.count_ef.keys():
            # uniform trans. prob. = count of this fe word / count of this f word
            self.t_ef[ef] = self.count_ef[ef] / self.total_f[ef.split('|')[-1]]
        sys.stderr.write("[uniform init] t(two|deux):" + str(self.t_ef['two|deux']) + "\n")

        # Loop until converge
        loop_count = 0
        while True:
            loop_count += 1
            converged = True
            # initialize
            for (n, (f, e)) in enumerate(self.bitext):
                for f_i in set(f):
                    self.total_f[f_i] = 0.0
                    for e_j in set(e):
                        ef = e_j + r'|' + f_i
                        self.count_ef[ef] = 0.0
            for (n, (f, e)) in enumerate(self.bitext):
                # compute normalization
                for e_j in set(e):
                    s_total = 0.0         # Normalization
                    for f_i in set(f):
                        ef = e_j + r'|' + f_i
                        s_total += self.t_ef[ef]
                # collect counts
                    for f_i in set(f):
                        ef = e_j + r'|' + f_i
                        count = self.t_ef[ef] / s_total
                        self.count_ef[ef] += count
                        self.total_f[f_i] += count
            # Estimate prob
            for (n, (f, e)) in enumerate(self.bitext):
                for f_i in f:
                    for e_j in e:
                        ef = e_j + r'|' + f_i
                        self.t_ef[ef] = self.count_ef[ef] / self.total_f[f_i]
            # check convergence
            for (n, (f, e)) in enumerate(self.bitext):
                for f_i in set(f):
                    for e_j in set(e):
                        ef = e_j + r'|' + f_i
                        if abs(self.t_ef[ef] - 1.0) < self.threshold or abs(self.t_ef[ef] - 0.0) < self.threshold:
                            pass
                        else:
                            converged = False
                            break
                if not converged:
                    break
            if converged:
                sys.stderr.write("training complete\n")
                break
            else:
                sys.stderr.write("t(two|deux):" + str(self.t_ef['two|deux']) + "\n")
                if loop_count % 10 == 0:
                    sys.stderr.write("loop #%d\n" % loop_count)


    def align(self):

        for (f, e) in self.bitext:
            for (i, f_i) in enumerate(f):
                for (j, e_j) in enumerate(e):
                    ef = e_j + r'|' + f_i
                    if self.t_ef[ef] >= self.threshold:
                        sys.stdout.write("%i-%i " % (i, j))
            sys.stdout.write("\n")

    def _isclose(self, a, b, rel_tol=0.25, abs_tol=0.0):
        """
        compare two float numbers
        :param a:
        :param b:
        :param rel_tol:the maximum allowed difference between a and b, relative to the larger absolute value of a or b.
        :param abs_tol:minimum absolute tolerance
        :return: boolean
        """
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)



