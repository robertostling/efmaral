# cython: profile=False
# cython: language_level=3

import numpy as np
from operator import itemgetter
import sys
import time
import random
import math

from gibbs import ibm_sample_parallel, ibm_initialize, ibm_create

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
FERT_ARRAY_LEN = 0x20


cdef class TokenizedText:
    """A tokenized text with indexed words

    Note that the NULL word is always represented by index 0, and an empty
    string in the voc tuple.

    sents -- sentences (tuple of ndarray[TOKEN_t])
    indexer -- mapping from strings to word indexes
    voc -- tuple of vocabulary, corresponding to indexer
    """

    cdef tuple sents
    cdef dict indexer
    cdef tuple voc

    def __init__(self, arg):
        """Create a new TokenizedText instance.

        If the argument is a str object, this is interpreted as a filename
        from which the file is read. If it is a list object, this is assumed
        to contain tokenized sentences as lists of strings.
        """

        if type(arg) is str: self.read_file(arg)
        elif type(arg) is list: self.read_sents(arg)

    cdef read_file(self, str filename):
        cdef list sents
        cdef str line
        with open(filename, 'r', encoding='utf-8') as f:
            sents = [line.lower().split() for line in f]
        self.read_sents(sents)

    cdef read_sents(self, list sents):
        cdef dict indexer
        cdef str s
        cdef list sent
        indexer = { '': 0 } # NULL word has index 0
        self.sents = tuple(
            np.array([indexer.setdefault(s, len(indexer)) for s in sent],
                     dtype=TOKEN_dtype)
            for sent in sents)
        self.indexer = indexer
        self.voc = tuple(
            s for s,_ in sorted(indexer.items(), key=itemgetter(1)))


cpdef read_fastalign(filename):
    """Read a file in fast_align format.

    Returns the two sides of the text as a tuple of two TokenizedText
    instances.
    """

    cdef list sents1, sents2
    cdef tuple pair
    cdef str line
    with open(filename, 'r', encoding='utf-8') as f:
        pair = tuple(zip(*[line.lower().split('|||') for line in f]))
    text1 = TokenizedText([line.split() for line in pair[0]])
    text2 = TokenizedText([line.split() for line in pair[1]])
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
                np.empty((ff.shape[0]*ee.shape[0],), dtype=COUNT_dtype)
                for ee,ff in zip(eee,fff))
        self.lex_n_len = ibm_create(
                eee, fff, self.lex_idx, len(e_voc), len(f_voc))

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
        aaa = tuple(np.empty_like(ff, dtype=LINK_dtype) for ff in self.fff)

        # Jump length counts, the priors are initialized here and the counts
        # by ibm_initialize(). The last value of the vector contains the sum
        # of all the other elements.
        jump_n = None if model < 2 else np.full(
                (JUMP_ARRAY_LEN+1,), 0.5, dtype=COUNT_dtype)
        if not jump_n is None:
            jump_n[-1] = jump_n[:-1].sum()

        # Fertility counts (per source type). The priors are initialized here
        # and the counts by ibm_initialize()
        fert_n = None if model < 3 else np.full(
                (FERT_ARRAY_LEN * len(self.e_voc),), 0.5, dtype=COUNT_dtype)

        # Initialize the five parameters, given the (constant) sentences and
        # vocabularies from both languages.
        ibm_initialize(self.eee, self.fff, aaa, self.lex_idx, lex_n, lex_n_sum,
                       jump_n, fert_n, len(self.e_voc), len(self.f_voc),
                       lex_alpha, null_alpha, seed, True)

        return (aaa, lex_n, lex_n_sum, jump_n, fert_n)


    cdef align(self,
               int prng_seed,
               int n_samplers,
               double null_prior,
               double lex_alpha,
               double null_alpha,
               tuple scheme):
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
        highest_model = max(model for model,n_epochs in scheme)

        print('Initializing %d sampler%s...' % (
            n_samplers, '' if n_samplers == 1 else 's'),
            file=sys.stderr)
        # Create parameters for n_samplers independent samplers.
        params = tuple(
                self.create_sampler(highest_model, lex_alpha, null_alpha, seed)
                for _ in range(n_samplers))

        # Create probability vectors where the final alignment distributions
        # will be stored. Each vector in the tuple corresponds to one sentence
        # pair, and consists of a flattened (E+1)*F array, for source and
        # target sentence lengths E and F.
        sent_ps = tuple(np.zeros(((ee.shape[0]+1)*ff.shape[0],),
                                 dtype=COUNT_dtype)
                        for ee,ff in zip(self.eee, self.fff))

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

        print('Computing final alignments...', file=sys.stderr)

        # Discretize the alignment distributions stored in sent_ps and return.
        return tuple(
            np.empty((0,), dtype=INDEX_dtype) \
            if ee.shape[0] == 0 or ff.shape[0] == 0
            else np.array(
                [NULL_LINK if a == ee.shape[0] else a for a in
                 ps.reshape((ff.shape[0], ee.shape[0]+1)).argmax(1)],
                dtype=INDEX_dtype)
            for ee, ff, ps in zip(self.eee, self.fff, sent_ps))


cpdef print_alignments(tuple aaa, bool reverse, f):
    """Write the given alignments to file f, using Moses format.

    aaa -- alignment variables, from Aligner.align()
    reverse -- invert source and target language order when writing alignments
    f -- file object to write to
    """

    cdef np.ndarray[INDEX_t, ndim=1] aa
    cdef int i, j
    cdef FILE* cfile
    cdef bool first

    cfile = fdopen(f.fileno(), 'w')
    for aa in aaa:
        first = True
        for i in range(aa.shape[0]):
            j = aa[i]
            if j != NULL_LINK:
                if not first: fputc(32, cfile)
                first = False
                if reverse: fprintf(cfile, b'%d-%d', i, j)
                else: fprintf(cfile, b'%d-%d', j, i)
        fputc(10, cfile)


def align(list filenames,
          int n_samplers,
          double length,
          double null_prior,
          double lex_alpha,
          double null_alpha,
          bool reverse,
          int seed):
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
    seed -- PRNG seed
    """

    cdef TokenizedText tt1, tt2
    cdef tuple voc1, voc2

    if len(filenames) == 1:
        print('Reading %s...' % filenames[0], file=sys.stderr)
        tt1, tt2 = read_fastalign(filenames[0])
    else:
        filename1, filename2 = filenames
        print('Reading %s...' % filename1, file=sys.stderr)
        tt1 = TokenizedText(filename1)
        print('Reading %s...' % filename2, file=sys.stderr)
        tt2 = TokenizedText(filename2)
    if reverse:
        tt1, tt2 = tt2, tt1

    if len(tt1.sents) != len(tt2.sents):
        raise ValueError('Source files have different number of sentences!')

    aligner = Aligner(tt1.voc, tt2.voc, tt1.sents, tt2.sents)
    n_samples = int(10000 / math.sqrt(len(tt1.sents)))

    # Scale by the user-supplied length parameter.
    n_samples = int(length * n_samples)
    # Impose absolute limits of 4 to 250 samples.
    # Also, it does not make sense to take fewer samples than we have parallel
    # samplers.
    n_samples = min(250, max(4, n_samplers, n_samples))

    print('Will collect %d samples.' % n_samples, file=sys.stderr)

    # The default scheme is to spend a third of the time going through
    # IBM1 and HMM, and the rest with the HMM+F model.
    scheme = ((1, n_samples/4), (2, n_samples/4), (3, n_samples))

    return aligner.align(seed, n_samplers, null_prior, lex_alpha, null_alpha,
                         scheme)

