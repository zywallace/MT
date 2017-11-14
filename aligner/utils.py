import optparse, sys

def open_file():
    # open data file using args from command line
    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
    optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
    optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
    optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float",
                         help="Threshold for aligning with Dice's coefficient (default=0.5)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int",
                         help="Number of sentences to use for training and alignment")
    (opts, _) = optparser.parse_args()
    f_data = "%s.%s" % (opts.train, opts.french)
    e_data = "%s.%s" % (opts.train, opts.english)
    return f_data,e_data,opts.num_sents
