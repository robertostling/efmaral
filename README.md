# efmaral
Efficient Markov Chain word alignment

`efmaral` is a tool for performing word alignment using Gibbs sampling with
a Bayesian extension of the IBM alignment models. It is **both** fast and
accurate. In addition, it can function as a plug-in replacement for
[fast_align](https://github.com/clab/fast_align), or be used as a library from
a Python program.

A thorough description can be found in
[Östling and Tiedemann (2016)](https://ufal.mff.cuni.cz/pbml/106/art-ostling-tiedemann.pdf) ([BibTeX](http://www.robos.org/sections/research/robert_bib.html#Ostling2016efmaral)).

## Installing

`efmaral` is implemented in Python/C and requires the following software to be
installed:

 * Python 3 (tested with version 3.4)
 * gcc (tested with version 4.9) and GNU Make
 * Cython (tested with version 0.21 and 0.20)
 * NumPY (tested with version 1.8.2)
 * `fast_align` (needed if you want to use the `atools` utility for symmetrization)

These can be installed on Debian 8 (jessie) using the following command as root:

    apt-get install python3-dev python3-setuptools cython3 gcc make python3-numpy

Then, clone this repository and run `make`:

    git clone https://github.com/robertostling/efmaral.git
    cd efmaral
    make

If everything works, you can try directly to evaluate with a small part of
the English-Hindi data set from WPT-05:

    python3 scripts/evaluate.py efmaral \
        3rdparty/data/test.eng.hin.wa \
        3rdparty/data/test.eng 3rdparty/data/test.hin \
        3rdparty/data/trial.eng 3rdparty/data/trial.hin

This uses the small trial set as training data, so the actual figures are
poor.


## Usage

For the lazy, there is a convenience script to align in both directions and
perform symmetrization. You can use it with the example data in the repo:

    scripts/align_symmetrize.sh 3rdparty/data/test.eng 3rdparty/data/test.hin test.moses grow-diag-final-and

The default values of `efmaral` should give acceptable results, but for a full
list of options, run:

    ./efmaral.py --help

Given a parallel text in `fast_align` format, `efmaral` can be used in the same
way as `fast_align`. First, we can convert some of the WPT-05 data above to
the `fast_align` format:

    python3 scripts/wpt2fastalign.py \
        3rdparty/data/test.eng 3rdparty/data/test.hin >test.fa

Then, we can run `efmaral` on this file:

    ./efmaral.py -i test.fa >test-fwd.moses 

If we want symmetrized alignments, also run in the reverse direction, then run
atools (which comes with `fast_align`) to symmetrize:

    ./efmaral.py -r -i test.fa >test-back.moses
    atools -i test-fwd.moses -j test-back.moses -c grow-diag-final-and >test.moses

`efmaral` also supports reading Europarl-style inputs directly, such as the
WPT data, by providing two filename arguments to the `-i` option:

    ./efmaral.py -i 3rdparty/data/test.eng 3rdparty/data/test.hin >test-fwd.moses


## Performance

This is a comparison between `efmaral` and `fast_align`.
Tests were run on a two-CPU
Xeon E5645 2.4 GHz machine, each CPU contains 6 cores with hyperthreading.
In all, 24 virtual cores are available.

The times given are the total runtime for the `evaluate.py` script, and
includes the (rather insignificant) time used by `evaluate.py` itself and the
`atools` symmetrization program.

Alignment Error Rate (AER) is used to indicate alignment accuracy (lower is
better). Both programs are run with the recommended parameters.

**Note:** the timings should be taken with a grain of salt for both systems,
since they depend critically on the number of iterations used, which could
easily be adjusted depending on the performance/accuracy ratio desired. Also
note that benchmarking on a 24-core system gives `fast_align` an advantage.

### efmaral

| Languages | Sentences | AER | CPU time (s) | Real time (s) |
| --------- | ---------:| ---:| ------------:| -------------:|
| English-Swedish | 1,862,426 | 0.133 | 1,719 | 620 |
| English-French | 1,130,551 | 0.085 | 763 | 279 |
| English-Inkutitut | 340,601 | 0.235 | 122 | 46 |
| Romanian-English | 48,681 | 0.287 | 161 | 46 |
| English-Hindi | 3,530 | 0.483 | 98 | 10 |

### fast_align

| Languages | Sentences | AER | CPU time (s) | Real time (s) |
| --------- | ---------:| ---:| ------------:| -------------:|
| English-Swedish | 1,862,426 | 0.205 | 11,090 | 672 |
| English-French | 1,130,551 | 0.153 | 3,840 | 241 |
| English-Inuktitut | 340,601 | 0.287 | 477 | 47 |
| Romanian-English | 48,681 | 0.325 | 208 | 17 |
| English-Hindi | 3,530 | 0.672 | 24 | 2 |

One advantage of the EM algorithm used by `fast_align` is that it is easy to
parallelize, unlike the collapsed Gibbs sampler `efmaral` uses. Therefore,
`fast_align` is sometimes faster when aligning a single text on a system with
many cores (such as this one). The *CPU time* column gives the total amount of
time used by all CPU cores, and should be relatively constant regardless of
the number of cores. In contrast, the *Real time* column gives the actual
running time on this 24-core system.

Note that `fast_align` uses a fixed number of iterations, whereas `efmaral`
tries to increase the number of sampling iterations when aligning small
corpora. Currently the number of iterations is proportional to the inverse of
the square root of the corpus length, but there is no deep analysis behind
this decision (nor a proper evaluation, so this may change in the future).

Both tools have an unusually high CPU time/real time ratio with the very
short English-Hindi corpus, this might be due to OpenMP overhead becoming
noticeable.


## Tips and tricks

The three most important options are probably:

 * `--length`: number of iterations relative to the number determined
   automatically (based on the number of sentences). The default value of 1.0
   means no change, whereas e.g. 0.2 would result in a 5x speedup (unless one
   hits the bottom number of 4 or top value of 250 iterations) -- probably at
   the cost of some accuracy.
 * `--null-prior`: prior probability of NULL alignment, this can to some
   extent be used to trade recall for precision, by setting a higher value
   than the default 0.2 (or vice versa, with a lower value).
 * `--samplers`: number of independent samplers to use. This reduces
   initialization bias, and is the only way for `efmaral` to utilize multiple
   CPU cores since each is run in a separate thread. The default is 2, but for
   maximum speed use a value of 1. A value much higher than 4 is probably not
   very useful.


## Aligning very large corpora

There are two things to consider: memory usage, and the index table pointer
size.

The amount of memory used is proportional to the sum of the products of the
lengths of each sentence pair, i.e. `\sum_{e,f} (|e| \times |f|)`.
In practice, this amounts to about 20 GB for aligning a large Europarl bitext
(such as the 1.86 million sentence English-Swedish corpus mentioned above).
If you align both directions in parallel, you need 40 GB.

If the parameter vector can not fit in an unsigned 32-bit integer,
it is necessary to change `INDEX_t` in `gibbs.c` as well as in `cyalign.pyx`.
This will however further increase memory usage,
so the code uses 32-bit integers by default.
An error will be printed if the parameter vector becomes too large.


## Implementation notes

The advantage of using Gibbs sampling is that, unlike
Expectation-Maximization (EM, used in tools such as `GIZA++`),
adding word order and fertility models do not affect the
computational complexity of inference.
`fast_align` circumvents this by using a variant of the much simpler Model 2,
but this reduces accuracy.

A naively implemented Gibbs sampler can however add a large constant factor,
which brings down performance.
The method used here to obtain an efficient sampler is to create a pre-computed
table indexing the (fixed) sequence of parameters accessed during each sampling
iteration, which reduces the inner loop of the sampler to a simple dot
product:

```c
    for (size_t i=0; i<ee_len; i++) {
        ps_sum += counts[counts_idx[i]] * counts_sum[ee[i]];
        ps[i] = ps_sum;
    }
```

The above is the actual code used for the IBM1 Gibbs sampler. Adding the HMM
word order model and a fertility model simply consists of adding two more
factors to the product (wihch are faster than the above, since they don't need
indirect lookups and random access in large arrays).

