# cython: profile=False
# cython: language_level=3

import numpy as np
from operator import itemgetter
import sys
import time
import random
import math

from cefmaral import *

cimport numpy as np
cimport cython
from cpython cimport bool
from libc.stdio cimport fprintf, fdopen, fputc, FILE

# These are also defined in gibbs.c, and must be the same!
# See gibbs.c for details.
ctypedef np.float32_t COUNT_t
COUNT_dtype = np.float32

ctypedef np.uint32_t INDEX_t
INDEX_dtype = np.uint32

ctypedef np.uint32_t TOKEN_t
TOKEN_dtype = np.uint32

ctypedef np.uint16_t LINK_t
LINK_dtype = np.uint16

ctypedef np.uint64_t PRNG_SEED_t
PRNG_SEED_dtype = np.uint64

# These constants are also defined separately in gibbs.c, must be the same!
NULL_LINK = 0xffff
JUMP_ARRAY_LEN = 0x800
FERT_ARRAY_LEN = 0x08


cdef class TokenizedText:
    """A tokenized text with indexed words

    Note that the NULL word is always represented by index 0, and an empty
    string in the voc tuple.

    sents -- sentences (tuple of ndarray[TOKEN_t])
    indexer -- mapping from strings to word indexes
    voc -- tuple of vocabulary, corresponding to indexer
    prefix_len -- if above 0, cut each word off after this many characters
    suffix_len -- same as above, but for suffixes rather than prefixes
    """

    cdef tuple sents
    cdef dict indexer
    cdef tuple voc
    cdef int prefix_len
    cdef int suffix_len

    def __init__(self, arg, prefix_len, suffix_len):
        """Create a new TokenizedText instance.

        If the argument is a str object, this is interpreted as a filename
        from which the file is read. If it is a list object, this is assumed
        to contain tokenized sentences as lists of strings.
        """
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len
        if type(arg) is str: self.read_file(arg)
        elif type(arg) is list: self.read_sents(arg)
        elif type(arg) is tuple:
            self.sents, self.voc, self.indexer = arg

    cdef read_file(self, str filename):
        cdef list sents
        cdef str line
        cdef str s
        with open(filename, 'r', encoding='utf-8') as f:
            sents = [line.lower().split() for line in f]
        self.read_sents(sents)

    cdef read_sents(self, list sents):
        cdef dict indexer
        cdef str s
        cdef list sent
        indexer = { '': 0 } # NULL word has index 0
        if self.prefix_len > 0:
            self.sents = tuple([
                np.array([indexer.setdefault(s[:self.prefix_len], len(indexer))
                          for s in sent],
                         dtype=TOKEN_dtype)
                for sent in sents])
        elif self.suffix_len > 0:
            self.sents = tuple([
                np.array([indexer.setdefault(s[-self.suffix_len:], len(indexer))
                          for s in sent],
                         dtype=TOKEN_dtype)
                for sent in sents])
        else:
            self.sents = tuple([
                np.array([indexer.setdefault(s, len(indexer)) for s in sent],
                         dtype=TOKEN_dtype)
                for sent in sents])
        self.indexer = indexer
        self.voc = tuple(
            [s for s,_ in sorted(indexer.items(), key=itemgetter(1))])


cpdef read_fastalign(filename, prefix_len, suffix_len):
    """Read a file in fast_align format.

    Returns the two sides of the text as a tuple of two TokenizedText
    instances.
    """

    cdef list sents1, sents2
    cdef tuple pair
    cdef str line
    with open(filename, 'r', encoding='utf-8') as f:
        pair = tuple(zip(*[line.lower().split('|||') for line in f]))
    text1 = TokenizedText([line.split() for line in pair[0]],
                          prefix_len, suffix_len)
    text2 = TokenizedText([line.split() for line in pair[1]],
                          prefix_len, suffix_len)
    return text1, text2


cdef class Aligner:
    cdef tuple e_voc            # source vocabulary
    cdef tuple f_voc            # target vocabulary
    cdef tuple eee              # source sentences
    cdef tuple fff              # target sentences
    cdef tuple lex_idx          # index for lexical counts + priors
    cdef int lex_n_len          # lenght of lexical counts array

    def __init__(self,
                 tuple e_voc,
                 tuple f_voc,
                 tuple eee,
                 tuple fff):
        """Initialize the aligned with vocabularies and priors, but no data.

        e_voc -- source language vocabulary (tuple of strings)
        f_voc -- target language vocabulary (tuple of strings)
        eee -- source language sentences (tuple of ndarray[TOKEN_t])
        fff -- target language sentences (tuple of ndarray[TOKEN_t])
        """
        self.e_voc = e_voc
        self.f_voc = f_voc
        self.eee = eee
        self.fff = fff

        print('Initializing aligner (%d sentences)...' % len(eee),
              file=sys.stderr)
        self.lex_idx = tuple(
                np.empty((ff.shape[0]*ee.shape[0],), dtype=INDEX_dtype)
                for ee,ff in zip(eee,fff))
        self.lex_n_len = ibm_create(
                eee, fff, self.lex_idx, len(e_voc), len(f_voc))
        print('Index vector contains %d elements.' % self.lex_n_len,
              file=sys.stderr)
        if INDEX_dtype == np.uint32 and self.lex_n_len >= 2**32:
            raise ValueError(
                    (('INDEX_t is 32-bit but index table size is %d! ' +
                      'See the README for instructions') %
                     self.lex_n_len))

    cdef tuple create_sampler(
            self,
            int model,
            double lex_alpha,
            double null_alpha,
            np.ndarray[PRNG_SEED_t, ndim=1] seed):
        """Initialize one sampler and return its parameters.

        model -- 1 for IBM1, 2 for HMM, 3 for HMM+Fertility
        lex_alpha -- Dirichlet prior for lexical distributions
        null_alpha -- Dirichlet prior for NULL word lexical distribution
        seed -- state of PRNG
        
        Returns a 5-tuple containing:
            - alignment variables (tuple of vectors, same shape as self.fff)
            - lexical counts
            - reciprocal sums of lexical counts
            - jump length counts
            - fertility counts
        """

        cdef tuple aaa
        cdef np.ndarray[COUNT_t, ndim=1] lex_n, lex_n_sum, jump_n, fert_n

        # Lexical counts, will be initialized by ibm_initialize()
        lex_n = np.empty((self.lex_n_len,), dtype=COUNT_dtype)

        # Reciprocal sums of lexical counts (per source type), will be
        # initialized by ibm_initialize()
        lex_n_sum = np.empty((len(self.e_voc),), dtype=COUNT_dtype)

        # Alignment variables, will be initialized by ibm_initialize()
        aaa = tuple([np.empty_like(ff, dtype=LINK_dtype) for ff in self.fff])

        # Jump length counts, the priors are initialized here and the counts
        # by ibm_initialize(). The last value of the vector contains the sum
        # of all the other elements.
        # TODO: need to modify gibbs.c since it keeps track of jump_n
        jump_n = np.full((JUMP_ARRAY_LEN+1,), 0.5, dtype=COUNT_dtype)
        #jump_n = None if model < 2 else np.full(
        #        (JUMP_ARRAY_LEN+1,), 0.5, dtype=COUNT_dtype)
        if not jump_n is None:
            jump_n[-1] = jump_n[:-1].sum()

        # Fertility counts (per source type). The priors are initialized here
        # and the counts by ibm_initialize()
        fert_n = None if model < 3 else np.full(
                (FERT_ARRAY_LEN * len(self.e_voc),), 0.5, dtype=COUNT_dtype)

        return (aaa, lex_n, lex_n_sum, jump_n, fert_n)


    cdef align(self,
               int prng_seed,
               int n_samplers,
               double null_prior,
               double lex_alpha,
               double null_alpha,
               tuple scheme,
               bool discretize):
        """Align the bitext this instance was created with.

        prng_seed -- seed for random state
        n_samplers -- number of indepnedent samplers
        null_prior -- prior for NULL word probability (between 0 and 1)
        scheme -- training scheme, containing tuples of (model, n_epochs)
        """

        cdef tuple params, sent_ps
        cdef int highest_model, model, n_epochs

        # Create a single random state which will be used throughout the
        # initialization and sampling procedure.
        random.seed(prng_seed)
        seed = np.array([random.getrandbits(64)], dtype=PRNG_SEED_dtype)

        # Find out what the highest model used is, we need to make sure that
        # all the parameters needed for this model are initialized.
        highest_model = max([model for model,n_epochs in scheme])

        print('Initializing %d sampler%s...' % (
            n_samplers, '' if n_samplers == 1 else 's'),
            file=sys.stderr)
        # Create (empty) parameter vectors for n_samplers independent samplers.
        params = tuple([
                self.create_sampler(highest_model, lex_alpha, null_alpha, seed)
                for _ in range(n_samplers)])
        # Initialize the parameters in parallel.
        ibm_initialize_parallel(
                params, self.eee, self.fff, self.lex_idx,
                len(self.e_voc), len(self.f_voc), lex_alpha, null_alpha,
                seed, True)

        # Create probability vectors where the final alignment distributions
        # will be stored. Each vector in the tuple corresponds to one sentence
        # pair, and consists of a flattened (E+1)*F array, for source and
        # target sentence lengths E and F.
        sent_ps = tuple([np.zeros(((ee.shape[0]+1)*ff.shape[0],),
                                  dtype=COUNT_dtype)
                         for ee,ff in zip(self.eee, self.fff)])

        # This is the main loop, going through each step of the training
        # scheme and calling ibm_sample_parallel() to do the actual job.
        for scheme_step, (model, n_epochs) in enumerate(scheme):
            model_name = ['', 'IBM1', 'HMM', 'HMM+F'][model]
            sys.stderr.write(model_name)
            sys.stderr.flush()
            t0 = time.time()
            # When the 5th parameter is not None, i.e. when we actually want
            # to use the sample, then the independent samplers will take turn
            # adding samples to sent_ps.
            # See gibbs.c for details.
            ibm_sample_parallel(
                    n_epochs, model, self.eee, self.fff,
                    None if scheme_step < len(scheme)-1 else sent_ps,
                    self.lex_idx, params, null_prior, seed)
            t = time.time() - t0
            print(' done (%.3f s)' % t, file=sys.stderr)

        # Borrow the sampling array from the first sampler, since we won't
        # need this anyway and it's already allocated.
        aaa = params[0][0]
        if discretize:
            print('Computing final alignments...', file=sys.stderr)
            ibm_discretize(sent_ps, aaa)
            return aaa
        else:
            return sent_ps


def align(list filenames,
          int n_samplers,
          double length,
          double null_prior,
          double lex_alpha,
          double null_alpha,
          bool reverse,
          int model,
          int prefix_len,
          int suffix_len,
          int seed,
          bool discretize=True,
          bool reshape=False):
    """Align the given file(s) and return the result.

    filenames -- a list of filenames, if it contains a single item it is
                 interpreted as a fast_align format file with both source and
                 target language sentences in the same file, otherwise as two
                 separate files with the same number of lines
    n_samplers -- number of parallel samplers
    length -- the number of sampling iterations is auto-determined based on
              file size, then multiplied by with value and rounded down to
              determine the actual number of iterations
    null_prior -- see Aligner.align()
    lex_alpha -- see Aligner.align()
    null_alpha -- see Aligner.align()
    reverse -- reverse the order of the source and target language when
               aligning
    model -- 1 for IBM1, 2 for HMM, 3 for HMM+fertility
    prefix_len -- 0 for no stemming, otherwise length of prefix to keep
    suffix_len -- 0 for no stemming, otherwise length of suffix to keep
    seed -- PRNG seed
    """

    cdef TokenizedText tt1, tt2
    cdef tuple voc1, voc2
    cdef int samples_min, samples_max

    if len(filenames) == 1:
        print('Reading %s...' % filenames[0], file=sys.stderr)
        tt1, tt2 = read_fastalign(filenames[0], prefix_len, suffix_len)
    else:
        filename1, filename2 = filenames
        print('Reading %s...' % filename1, file=sys.stderr)
        tt1 = TokenizedText(filename1, prefix_len, suffix_len)
        print('Reading %s...' % filename2, file=sys.stderr)
        tt2 = TokenizedText(filename2, prefix_len, suffix_len)
    if reverse:
        tt1, tt2 = tt2, tt1

    print('Vocabulary size: %d (source), %d (target)' % (
            len(tt1.voc), len(tt2.voc)),
          file=sys.stderr)

    index_size = sum(sent1.shape[0] * sent2.shape[0]
                     for sent1, sent2 in zip(tt1.sents, tt2.sents))
    print('Index table will require %d elements.' % index_size,
          file=sys.stderr)

    if len(tt1.sents) != len(tt2.sents):
        raise ValueError('Source files have different number of sentences!')

    aligner = Aligner(tt1.voc, tt2.voc, tt1.sents, tt2.sents)
    n_samples = int(10000 / math.sqrt(len(tt1.sents)))

    # Scale by the user-supplied length parameter.
    n_samples = int(length * n_samples)
    # Impose absolute limits of 4 to 250 samples.
    # Also, it does not make sense to take fewer samples than we have parallel
    # samplers.
    samples_max = max(1, int(250*length))
    samples_min = max(1, int(4*length))
    n_samples = min(samples_max, max(samples_min, n_samplers, n_samples))

    print('Will collect %d samples.' % n_samples, file=sys.stderr)

    # The default scheme is to spend a third of the time going through
    # IBM1 and HMM, and the rest with the HMM+F model.
    if model == 1:
        scheme = ((1, n_samples),)
    elif model == 2:
        scheme = ((1, max(1, n_samples//4)), (2, n_samples))
    else:
        scheme = ((1, max(1, n_samples//4)), (2, max(1, n_samples//4)),
                  (3, n_samples))

    aaa = aligner.align(seed, n_samplers, null_prior, lex_alpha, null_alpha,
                        scheme, discretize)

    if reshape and not discretize:
        def normalize(m):
            return m / m.sum(axis=1)[:,None]
        return tuple(normalize(aa.reshape(
                        aligner.fff[i].shape[0], aligner.eee[i].shape[0]+1))
                     for i, aa in enumerate(aaa))

    return aaa


def align_soft(
        list sents1,
        list sents2,
        int n_samplers,
        double length,
        double null_prior,
        double lex_alpha,
        double null_alpha,
        bool reverse,
        int model,
        int prefix_len1,
        int suffix_len1,
        int prefix_len2,
        int suffix_len2,
        int seed):
    """Align the given file(s) and return alignment marginal distributions.

    See align() for further information.
    This function accepts lists with words (tokenized sentences) instead of
    filenames like align().
    """

    cdef TokenizedText tt1, tt2
    cdef tuple voc1, voc2
    cdef int samples_min, samples_max

    tt1 = TokenizedText(sents1, prefix_len1, suffix_len1)
    tt2 = TokenizedText(sents2, prefix_len2, suffix_len2)

    index_size = sum(sent1.shape[0] * sent2.shape[0]
                     for sent1, sent2 in zip(tt1.sents, tt2.sents))
    print('Index table will require %d elements.' % index_size,
          file=sys.stderr)

    if len(tt1.sents) != len(tt2.sents):
        raise ValueError('Trying to align texts of different lengths!')

    aligner = Aligner(tt1.voc, tt2.voc, tt1.sents, tt2.sents)
    n_samples = int(10000 / math.sqrt(len(tt1.sents)))

    # Scale by the user-supplied length parameter.
    n_samples = int(length * n_samples)
    # Impose absolute limits of 4 to 250 samples.
    # Also, it does not make sense to take fewer samples than we have parallel
    # samplers.
    samples_max = max(1, int(250*length))
    samples_min = max(1, int(4*length))
    n_samples = min(samples_max, max(samples_min, n_samplers, n_samples))

    print('Will collect %d samples.' % n_samples, file=sys.stderr)

    # The default scheme is to spend a third of the time going through
    # IBM1 and HMM, and the rest with the HMM+F model.
    if model == 1:
        scheme = ((1, n_samples),)
    elif model == 2:
        scheme = ((1, max(1, n_samples//4)), (2, n_samples))
    else:
        scheme = ((1, max(1, n_samples//4)), (2, max(1, n_samples//4)),
                  (3, n_samples))

    return aligner.align(seed, n_samplers, null_prior, lex_alpha, null_alpha,
                         scheme, False)


def align_numeric(
        tuple sents1,
        tuple sents2,
        tuple voc1,
        tuple voc2,
        dict indexer1,
        dict indexer2,
        int n_samplers,
        double length,
        double null_prior,
        double lex_alpha,
        double null_alpha,
        bool reverse,
        int model,
        int seed,
        bool discretize):
    """Align the given file(s) and return alignment marginal distributions.

    See align() for further information.
    This function accepts lists with ndarray(uint32) for sentences.
    """

    cdef TokenizedText tt1, tt2
    cdef int samples_min, samples_max

    if reverse:
        sents1, sents2 = sents2, sents1
        voc1, voc2 = voc2, voc1
        indexer1, indexer2 = indexer2, indexer1

    tt1 = TokenizedText((sents1, voc1, indexer1), 0, 0)
    tt2 = TokenizedText((sents2, voc2, indexer2), 0, 0)

    index_size = sum(sent1.shape[0] * sent2.shape[0]
                     for sent1, sent2 in zip(tt1.sents, tt2.sents))
    print('Index table will require %d elements.' % index_size,
          file=sys.stderr)

    if len(tt1.sents) != len(tt2.sents):
        raise ValueError('Trying to align texts of different lengths!')

    aligner = Aligner(tt1.voc, tt2.voc, tt1.sents, tt2.sents)
    n_samples = int(10000 / math.sqrt(len(tt1.sents)))

    # Scale by the user-supplied length parameter.
    n_samples = int(length * n_samples)
    # Impose absolute limits of 4 to 250 samples.
    # Also, it does not make sense to take fewer samples than we have parallel
    # samplers.
    samples_max = max(1, int(250*length))
    samples_min = max(1, int(4*length))
    n_samples = min(samples_max, max(samples_min, n_samplers, n_samples))

    print('Will collect %d samples.' % n_samples, file=sys.stderr)

    # The default scheme is to spend a third of the time going through
    # IBM1 and HMM, and the rest with the HMM+F model.
    if model == 1:
        scheme = ((1, n_samples),)
    elif model == 2:
        scheme = ((1, max(1, n_samples//4)), (2, n_samples))
    else:
        scheme = ((1, max(1, n_samples//4)), (2, max(1, n_samples//4)),
                  (3, n_samples))

    return aligner.align(seed, n_samplers, null_prior, lex_alpha, null_alpha,
                         scheme, discretize)


cpdef tuple get_link_coo(tuple eee_fff, tuple aaa, int e_base, int f_base):
    cdef int n_links, n_f, n_e, i, j
    cdef np.ndarray[TOKEN_t, ndim=1] ee, ff
    cdef np.ndarray[LINK_t, ndim=1] aa
    cdef np.ndarray[TOKEN_t, ndim=1] ii
    cdef np.ndarray[TOKEN_t, ndim=1] jj

    n_links = 0
    for i in range(len(eee_fff)):
        ee, ff = eee_fff[i]
        aa = aaa[i]
        n_f = len(ff)
        n_e = len(ee)
        j = 0
        while j < n_f:
            a = aa[j]
            if a < n_e:
                n_links += 1
            j += 1

    ii = np.empty((n_links,), dtype=TOKEN_dtype)
    jj = np.empty((n_links,), dtype=TOKEN_dtype)

    n_links = 0
    for i in range(len(eee_fff)):
        ee, ff = eee_fff[i]
        aa = aaa[i]
        n_f = len(ff)
        n_e = len(ee)
        j = 0
        while j < n_f:
            a = aa[j]
            if a < n_e:
                f = ff[j]
                e = ee[a]
                ii[n_links] = e_base + e
                jj[n_links] = f_base + f
                n_links += 1
            j += 1

    return ii, jj

