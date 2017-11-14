Usage: `python decoder.py -n 10` will translate the first 10 french sentences.

Team member: Yu Zhao, Fan Yang, Zikun Chen

Implemented a stack-based beam search decoder that supports global reordering.

Implemented a [greedy decoder](http://www.iro.umontreal.ca/~felipe/bib2webV0.81/cv/papers/paper-tmi-2007.pdf). We used the result of beam search decoder as the seed and modified some transform methods.
