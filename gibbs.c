#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <numpy/npy_common.h>
#include <stdint.h>
#include <stdio.h>

#include "intmap.c"

// Store reciprocals of lexical distribution normalization factors
// This is much faster, but for large corpora there may be issues with
// numerical stability (although I haven't encountered any such case yet).
#define CACHE_RECIPROCAL    1

// This should be higher than the length of most sentences.
// Changing this requires updating cyalign.pyx!
#define JUMP_ARRAY_LEN  0x800
#define JUMP_SUM        JUMP_ARRAY_LEN

// Changing this requires updating cyalign.pyx!
#define FERT_ARRAY_LEN  0x20

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

// Note that these data types are defined separately in cylign.pyx, and must
// have the same value there as here!
typedef uint16_t LINK_t;        // type of alignment variables
typedef uint32_t TOKEN_t;       // type of tokens
typedef float COUNT_t;          // type of almost all floating-point values
#define NPY_COUNT NPY_FLOAT32   // must be the same as above!
typedef uint32_t INDEX_t;       // type of indexes for the lexical counts
                                // vector, may need to be increased for some
                                // huge corpora
typedef uint64_t PRNG_SEED_t;   // PRNG state

// Value to represent links to the NULL word internally
static const LINK_t null_link = 0xffff;

// Xorshift* PRNG (from Wikipedia)
inline static PRNG_SEED_t prng_next(PRNG_SEED_t x) {
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    return x * 2685821657736338717ULL;
}

inline static COUNT_t prng_next_count(PRNG_SEED_t *seed) {
    const PRNG_SEED_t x = (*seed = prng_next(*seed));
    return (COUNT_t)x / (COUNT_t)18446744073709551616.0;
}

inline static int prng_next_int(PRNG_SEED_t *seed, int m) {
    const PRNG_SEED_t x = (*seed = prng_next(*seed));
    return (int)(x % (PRNG_SEED_t)m);
}

// Get the index of the jump distribution parameter vector for a jump from
// position i to j (in a sentence with total length len)
inline static size_t get_jump_index(int i, int j, int len) {
    return (size_t)MAX(0, MIN(JUMP_ARRAY_LEN-1, j - i + JUMP_ARRAY_LEN/2));
}

inline static size_t get_fert_index(size_t e, int fert) {
    //if (fert < 0) return FERT_ARRAY_LEN-1;
    return e*FERT_ARRAY_LEN + (size_t)MIN(fert, FERT_ARRAY_LEN-1);
}

// Draw a random sample from an unnormalized cumulative categorical
// distribution.
inline static size_t categorical_sample(
    const COUNT_t *ps,  // ps[i] is proportional to sum(p_0, p_1, ... p_i)
    size_t ps_len,      // Length of ps array
    COUNT_t r)          // Random number between 0 and 1
{
    r *= ps[ps_len-1];
    for (size_t i=0; i<ps_len-1; i++)
        if (ps[i] >= r) return i;
    return ps_len-1;
}


// Sample alignments for one sentence
static void gibbs_ibm_sample(
    int model,                  // model (1 = IBM1, 2 = HMM, 3 = HMM+F)
    const TOKEN_t *ee,          // source sentence
    size_t ee_len,              // length of source sentence
    const TOKEN_t *ff,          // target sentence
    size_t ff_len,              // length of target sentence
    LINK_t *aa,                 // alignment variables to sample
    COUNT_t *dists,             // buffer to add normalized sampling
                                // distributions to (or NULL to skip adding)
    const INDEX_t *counts_idx,  // indexes into counts array
    COUNT_t *counts,            // lexical priors + counts
    COUNT_t *counts_sum,        // reciprocal sums of counts
    COUNT_t *jump_counts,       // jump length priors + counts
    COUNT_t *fert_counts,       // fertility priors + counts
    COUNT_t null_prior,         // prior for NULL alignment
    PRNG_SEED_t *seed)          // random state (will be modified)
{
    // Buffer for the sampling distributions
    COUNT_t ps[ee_len+1];

    // Fertility count for each token in the source sentence
    int fert[ee_len];
    for (size_t i=0; i<ee_len+1; i++)
        fert[i] = 0;
    for (size_t j=0; j<ff_len; j++)
        if (aa[j] != null_link) fert[aa[j]]++;

    // aa_jp1_table[j] will contain the alignment of the nearest non-NULL
    // aligned word to the right (or ee_len if there is no such word)
    int aa_jp1_table[ff_len];

    int aa_jp1 = ee_len;
    for (size_t j=ff_len; j>0; j--) {
        aa_jp1_table[j-1] = aa_jp1;
        if (aa[j-1] != null_link) aa_jp1 = aa[j-1];
    }

    // aa_jm1 will always contain the alignment of the nearest non-NULL
    // aligned word to the left (or -1 if there is no such word)
    int aa_jm1 = -1;
    for (size_t j=0; j<ff_len; j++) {
        aa_jp1 = aa_jp1_table[j];

        // Subtract lexical and fertility counts for the current alignment
        // of ff[j]
        size_t old_e;
        if (aa[j] == null_link) {
            counts[ff[j]] -= (COUNT_t) 1.0;
            old_e = 0;
        } else {
            const size_t old_i = aa[j];
            counts[counts_idx[old_i]] -= (COUNT_t) 1.0;
            old_e = (size_t)ee[old_i];
            fert_counts[get_fert_index(old_e, fert[old_i])] -= (COUNT_t) 1.0;
            fert[old_i]--;
            fert_counts[get_fert_index(old_e, fert[old_i])] += (COUNT_t) 1.0;
        }

#if CACHE_RECIPROCAL
        counts_sum[old_e] = (COUNT_t)1.0 /
                            ((COUNT_t)1.0/counts_sum[old_e] - (COUNT_t)1.0);
#else
        counts_sum[old_e] -= (COUNT_t)1.0;
#endif

        // This is the only jump done if ff[j] is NULL aligned
        const size_t skip_jump = get_jump_index(aa_jm1, aa_jp1, ee_len);

        // Remove jump length counts from the current alignment of ff[j]
        if (aa[j] == null_link) {
            jump_counts[JUMP_SUM] -= (COUNT_t) 1.0;
            jump_counts[skip_jump] -= (COUNT_t) 1.0;
        } else {
            const size_t old_jump1 = get_jump_index(aa_jm1, aa[j], ee_len);
            const size_t old_jump2 = get_jump_index(aa[j], aa_jp1, ee_len);

            jump_counts[JUMP_SUM] -= (COUNT_t) 2.0;
            jump_counts[old_jump1] -= (COUNT_t) 1.0;
            jump_counts[old_jump2] -= (COUNT_t) 1.0;
        }

        // ps_sum keeps the unnormalized cumulative probability of the
        // sampling distribution, and at the end of the sampling loop will
        // contain the inverse normalization factor
        COUNT_t ps_sum = (COUNT_t) 0.0;

        if (model == 1) {
            // For IBM1, only consider lexical distributions
            for (size_t i=0; i<ee_len; i++) {
#if CACHE_RECIPROCAL
                ps_sum += counts[counts_idx[i]] * counts_sum[ee[i]];
#else
                ps_sum += counts[counts_idx[i]] / counts_sum[ee[i]];
#endif
                ps[i] = ps_sum;
            }
        } else if (model == 2) {
            // For HMM model, also consider jump length distributions
            size_t jump1 = get_jump_index(aa_jm1, 0, ee_len);
            size_t jump2 = get_jump_index(0, aa_jp1, ee_len);

            for (size_t i=0; i<ee_len; i++) {
#if CACHE_RECIPROCAL
                ps_sum += (counts[counts_idx[i]] * counts_sum[ee[i]] *
                           jump_counts[jump1] * jump_counts[jump2]);
#else
                ps_sum += (counts[counts_idx[i]] / counts_sum[ee[i]] *
                           jump_counts[jump1] * jump_counts[jump2]);
#endif
                ps[i] = ps_sum;
                // We can same a few cycles by replacing calls to
                // get_jump_index() with bounded increment/decrement
                // operations of the jump length distribution indexes
                jump1 = MIN(JUMP_ARRAY_LEN-1, jump1+1);
                jump2 = MAX(0, jump2-1);
            }
        } else {
            // For HMM+Fertility model, consider lexical, jump length and
            // fertility distributions.
            size_t jump1 = get_jump_index(aa_jm1, 0, ee_len);
            size_t jump2 = get_jump_index(0, aa_jp1, ee_len);

            for (size_t i=0; i<ee_len; i++) {
                const size_t fert_idx = get_fert_index(ee[i], fert[i]);
                // The change in total probability by replacing the current
                // fertility of source word ee[i] with a value one higher.
                // If this word already has reached the maximum fertility
                // (FERT_ARRAY_LEN), a very unlikely scenario, assume no
                // change in probability.
                const COUNT_t fert_p_ratio =
                    (fert_idx == FERT_ARRAY_LEN - 1) ? 1.0
                    : fert_counts[fert_idx+1] / fert_counts[fert_idx];

#if CACHE_RECIPROCAL
                ps_sum += (counts[counts_idx[i]] * counts_sum[ee[i]] *
                           jump_counts[jump1] * jump_counts[jump2] *
                           fert_p_ratio);
#else
                ps_sum += (counts[counts_idx[i]] / counts_sum[ee[i]] *
                           jump_counts[jump1] * jump_counts[jump2] *
                           fert_p_ratio);
#endif
                ps[i] = ps_sum;
                jump1 = MIN(JUMP_ARRAY_LEN-1, jump1+1);
                jump2 = MAX(0, jump2-1);
            }
        }

        // NULL generation probability
        // While this only depends on the lexical parameter, for the HMM and
        // HMM+F models we need to use different normalization factors.
        if (model == 1) {
#if CACHE_RECIPROCAL
            ps_sum += null_prior * counts[ff[j]] * counts_sum[0];
#else
            ps_sum += null_prior * counts[ff[j]] / counts_sum[0];
#endif
        } else {
        // NOTE: rather than scaling the non-NULL probabilities with Z^-2 for
        // the jump distribution normalization factor Z, we scale the NULL
        // probability with Z^1 instead, since the sampling distribution will
        // be normalized anyway.
        // Beware of this if you ever make modifications here!
#if CACHE_RECIPROCAL
            ps_sum += null_prior * counts[ff[j]] * counts_sum[0] *
                      jump_counts[JUMP_SUM] * jump_counts[skip_jump];
#else
            ps_sum += null_prior * counts[ff[j]] / counts_sum[0] *
                      jump_counts[JUMP_SUM] * jump_counts[skip_jump];
#endif
        }
        // Write probability of NULL alignment
        ps[ee_len] = ps_sum;

        // Add normalized sampling distribution to the buffer
        if (dists != NULL) {
            const COUNT_t ps_scale = (COUNT_t)1.0 / ps_sum;
            // Since the cumulative distribution is stored, we need to
            // subtract neighboring values when normalizing
            dists[0] += ps[0] * ps_scale;
            for (size_t i=1; i<ee_len+1; i++)
                dists[i] += (ps[i] - ps[i-1]) * ps_scale;
            dists += ee_len+1;
        }

        // Sample from the distribution
        const COUNT_t r = prng_next_count(seed);
        const size_t new_i = categorical_sample(ps, ee_len+1, r);

        // Add lexical and fertility counts from the new alignment
        size_t new_e;
        if (new_i == ee_len) {
            aa[j] = null_link;
            counts[ff[j]] += (COUNT_t) 1.0;
            new_e = 0;
        } else {
            aa[j] = (LINK_t) new_i;
            counts[counts_idx[new_i]] += (COUNT_t) 1.0;
            new_e = ee[new_i];
            fert_counts[get_fert_index(new_e, fert[new_i])] -= (COUNT_t) 1.0;
            fert[new_i]++;
            fert_counts[get_fert_index(new_e, fert[new_i])] += (COUNT_t) 1.0;
        }

#if CACHE_RECIPROCAL
        counts_sum[new_e] = (COUNT_t)1.0 /
                            ((COUNT_t)1.0/counts_sum[new_e] + (COUNT_t)1.0);
#else
        counts_sum[new_e] += (COUNT_t)1.0;
#endif

        // Add jump length counts for the new alignment
        if (new_i == ee_len) {
            jump_counts[JUMP_SUM] += (COUNT_t) 1.0;
            jump_counts[skip_jump] += (COUNT_t) 1.0;
        } else {
            const size_t new_jump1 = get_jump_index(aa_jm1, new_i, ee_len);
            const size_t new_jump2 = get_jump_index(new_i, aa_jp1, ee_len);

            jump_counts[JUMP_SUM] += (COUNT_t) 2.0;
            jump_counts[new_jump1] += (COUNT_t) 1.0;
            jump_counts[new_jump2] += (COUNT_t) 1.0;

            aa_jm1 = new_i;
        }

        // Advance in the indexing vector for the lexical counts array
        counts_idx += ee_len;
    }
}

// Sample in parallel from a number of samplers using the same text
//
// See docstring at the Python definition below for details.
static PyObject *py_gibbs_ibm_sample_parallel(PyObject *self, PyObject *args) {
    PyObject *eee, *fff, *counts_idx_arrays, *dists_arrays, *params;
    PyArrayObject *seed_array;
    int model, n_iterations;
    double null_prior;

    if(!PyArg_ParseTuple(args, "iiOOOOOdO",
        &n_iterations, &model, &eee, &fff, &dists_arrays, &counts_idx_arrays,
        &params, &null_prior, &seed_array))
        return NULL;

    const size_t n_sents = PyTuple_Size(eee);
    const size_t n_samplers = PyTuple_Size(params);

    PRNG_SEED_t *seed = (PRNG_SEED_t*) PyArray_GETPTR1(seed_array, 0);
    PRNG_SEED_t local_seeds[n_samplers];
    for (size_t sampler=0; sampler<n_samplers; sampler++)
        local_seeds[sampler] = prng_next(*seed + 1 + sampler);
    *seed = prng_next(*seed);

    for (size_t iteration=0; iteration<n_iterations; iteration++) {
        fputc('.', stderr);
        fflush(stderr);
#pragma omp parallel for
        for (size_t sampler=0; sampler<n_samplers; sampler++) {
            // Skip sampling if we don't need the result.
            if (iteration > n_iterations-n_samplers+sampler) continue;

            // Each sampler gets to add to the overall result in turn.
            // If add_sample is 0, only the parameters (in the params tuple)
            // will be updated, otherwise the sampled distributions will be
            // added to dists_arrays.
            const int add_sample = (dists_arrays != Py_None) &&
                                   ((iteration % n_samplers) == sampler);

            // Thread-local cache for the PRNG seed (not sure if this actually
            // brings any performance advantage).
            PRNG_SEED_t seed_cache = local_seeds[sampler];

            PyObject *sampler_params = PyTuple_GET_ITEM(params, sampler);

            PyObject *aaa = PyTuple_GET_ITEM(sampler_params, 0);
            PyArrayObject *counts_array =
                (PyArrayObject*) PyTuple_GET_ITEM(sampler_params, 1);
            PyArrayObject *counts_sum_array =
                (PyArrayObject*) PyTuple_GET_ITEM(sampler_params, 2);
            PyArrayObject *jump_counts_array =
                (PyArrayObject*) PyTuple_GET_ITEM(sampler_params, 3);
            PyArrayObject *fert_counts_array =
                (PyArrayObject*) PyTuple_GET_ITEM(sampler_params, 4);

            COUNT_t *counts_sum =
                 (COUNT_t*) PyArray_GETPTR1(counts_sum_array, 0);
            COUNT_t *counts = (COUNT_t*) PyArray_GETPTR1(counts_array, 0);
            COUNT_t *jump_counts = ((PyObject*)jump_counts_array == Py_None)
                ? NULL
                : PyArray_GETPTR1(jump_counts_array, 0);
            COUNT_t *fert_counts = ((PyObject*)fert_counts_array == Py_None)
                ? NULL
                : PyArray_GETPTR1(fert_counts_array, 0);

            for (size_t sent=0; sent<n_sents; sent++) {
                PyArrayObject *ee_array =
                    (PyArrayObject*) PyTuple_GET_ITEM(eee, sent);
                PyArrayObject *ff_array =
                    (PyArrayObject*) PyTuple_GET_ITEM(fff, sent);
                const size_t ee_len = (size_t) PyArray_DIM(ee_array, 0);
                const size_t ff_len = (size_t) PyArray_DIM(ff_array, 0);

                if (ee_len == 0 || ff_len == 0) continue;

                PyArrayObject *aa_array =
                    (PyArrayObject*) PyTuple_GET_ITEM(aaa, sent);
                PyArrayObject *dists_array = (PyArrayObject*) (
                    (!add_sample)
                    ? NULL
                    : PyTuple_GET_ITEM(dists_arrays, sent));
                PyArrayObject *counts_idx_array =
                    (PyArrayObject*) PyTuple_GET_ITEM(counts_idx_arrays, sent);

                const TOKEN_t *ee =
                    (const TOKEN_t*) PyArray_GETPTR1(ee_array, 0);
                const TOKEN_t *ff =
                    (const TOKEN_t*) PyArray_GETPTR1(ff_array, 0);
                LINK_t *aa = (LINK_t*) PyArray_GETPTR1(aa_array, 0);
                COUNT_t *dists = (dists_array == NULL)
                    ? NULL
                    : PyArray_GETPTR1(dists_array, 0);
                const INDEX_t *counts_idx = (const INDEX_t*) PyArray_GETPTR1(
                    counts_idx_array, 0);

                gibbs_ibm_sample(model, ee, ee_len, ff, ff_len, aa, dists,
                                 counts_idx, counts, counts_sum,
                                 jump_counts, fert_counts, (COUNT_t)null_prior,
                                 &seed_cache);
            }

            local_seeds[sampler] = seed_cache;
        } // end of parallel section
    }

    Py_INCREF(Py_None);
    return Py_None;
}


// Initialize common parameters for a bitext (not individual samplers).
//
// See docstring at the Python definition below for details.
static PyObject *py_gibbs_ibm_create(PyObject *self, PyObject *args) {
    PyObject *eee, *fff, *counts_idx_arrays;
    unsigned long e_voc_size, f_voc_size;

    if(!PyArg_ParseTuple(args, "OOOkk",
                         &eee, &fff, &counts_idx_arrays,
                         &e_voc_size, &f_voc_size))
        return NULL;

    // This maps pairs of (e,f), coded as (uint32_t)(e*f_voc_size + f),
    // to indexes in the lexical counts array (which each independent sampler
    // has one).
    intmap counts_idx_map;
    intmap_create(&counts_idx_map, 0x1000);
    // counts_len contains the number of indexes in the lexical counts array
    // reserved so far.
    size_t counts_len = 0;

    size_t n_sents = PyTuple_Size(eee);

    // Reserve space for NULL generation probabilities
    // This is a special case, because it's known to be dense. Starting after
    // f_voc_size items, pairs (e,f) indexed by the sparse counts_idx_map are
    // used.
    //
    // For instance, with the toy corpus A B A ||| X Y Z, the lexical counts
    // array will have the following structure:
    //
    // 0    NULL-X
    // 1    NULL-Y
    // 2    NULL-Z
    // 3    A-X
    // 4    A-Y
    // 5    A-Z
    // 6    B-X
    // 7    B-Y
    // 8    B-Z
    //
    // And the the counts_idx array for this (only) sentence pair will
    // contain:
    //
    // 3 4 5 6 7 8 3 4 5
    //
    // Corresponding to the alignments considered by the sampler (i.e.
    // A-X, A-Y, A-Z, B-X, B-Y, B-Z, A-X, A-Y, A-Z -- note that NULL alignment
    // options are not included in the index)
    counts_len += f_voc_size;

    for (size_t sent=0; sent<n_sents; sent++) {
        PyArrayObject *ee_array = (PyArrayObject*) PyTuple_GET_ITEM(eee, sent);
        PyArrayObject *ff_array = (PyArrayObject*) PyTuple_GET_ITEM(fff, sent);
        const size_t ee_len = (size_t) PyArray_DIM(ee_array, 0);
        const size_t ff_len = (size_t) PyArray_DIM(ff_array, 0);

        if (ee_len == 0 || ff_len == 0) continue;

        PyArrayObject *counts_idx_array =
            (PyArrayObject*) PyTuple_GET_ITEM(counts_idx_arrays, sent);

        const TOKEN_t *ee = (const TOKEN_t*) PyArray_GETPTR1(ee_array, 0);
        const TOKEN_t *ff = (const TOKEN_t*) PyArray_GETPTR1(ff_array, 0);
        INDEX_t *counts_idx = (INDEX_t*) PyArray_GETPTR1(counts_idx_array, 0);

        for (size_t j=0; j<ff_len; j++) {
            const TOKEN_t f = ff[j];
            for (size_t i=0; i<ee_len; i++) {
                const TOKEN_t e = ee[i];
                // idx is a unique integer for each pair of source/target
                // token
                uint32_t idx = (uint32_t)f*(uint32_t)e_voc_size + (uint32_t)e;
                // ptr is the index within the lexical counts vector of idx,
                // this is found by hash table lookup
                // If it is not already in the hash table, reserve a new index
                // in the lexical counts vector for it.
                INDEX_t ptr =
                    intmap_setdefault(&counts_idx_map, idx, counts_len);
                if (ptr == counts_len) counts_len++;
                *(counts_idx++) = ptr;
            }
        }
    }

    intmap_free(&counts_idx_map);

    // Return the length of the lexical counts vector, because the caller will
    // want to create one (or several) and initialize them with
    // py_gibbs_ibm_initialize()
    return PyLong_FromSize_t(counts_len);
}

// Discretize the soft alignments created by a sampler.
static PyObject *py_gibbs_ibm_discretize(PyObject *self, PyObject *args) {
    PyObject *dists_arrays, *aaa;

    if(!PyArg_ParseTuple(args, "OO", &dists_arrays, &aaa)) return NULL;

    const size_t n_sents = PyTuple_Size(aaa);

#pragma omp parallel for
    for (size_t sent=0; sent<n_sents; sent++) {
        PyArrayObject *aa_array =
            (PyArrayObject*) PyTuple_GET_ITEM(aaa, sent);
        PyArrayObject *dists_array =
            (PyArrayObject*) PyTuple_GET_ITEM(dists_arrays, sent);
        const size_t ff_len =
            (size_t) PyArray_DIM(aa_array, 0);
        if (ff_len == 0) continue;

        const size_t ee_len =
            ((size_t) PyArray_DIM(dists_array, 0) / ff_len) - 1;
        if (ee_len == 0) continue;

        LINK_t *aa = (LINK_t*) PyArray_GETPTR1(aa_array, 0);
        COUNT_t *ps = (COUNT_t*) PyArray_GETPTR1(dists_array, 0);

        for (size_t j=0; j<ff_len; j++) {
            COUNT_t max_p = (COUNT_t) ps[0];
            size_t argmax_i = 0;
            for (size_t i=1; i<ee_len+1; i++) {
                if (ps[i] > max_p) {
                    max_p = ps[i];
                    argmax_i = i;
                }
            }
            aa[j] = (argmax_i == ee_len)? null_link : argmax_i;
            ps += ee_len + 1;
        }
    }

    Py_INCREF(Py_None);
    return Py_None;
}


// Print alignments to file
static PyObject *py_gibbs_ibm_print(PyObject *self, PyObject *args) {
    PyObject *aaa;
    int reverse, fd;

    if(!PyArg_ParseTuple(args, "Opi", &aaa, &reverse, &fd)) return NULL;

    const size_t n_sents = PyTuple_Size(aaa);
    FILE *file = fdopen(fd, "w");

    for (size_t sent=0; sent<n_sents; sent++) {
        PyArrayObject *aa_array =
            (PyArrayObject*) PyTuple_GET_ITEM(aaa, sent);
        const size_t aa_len = (size_t) PyArray_DIM(aa_array, 0);
        if (aa_len > 0) {
            LINK_t *aa = (LINK_t*) PyArray_GETPTR1(aa_array, 0);
            int first = 1;
            for (size_t j=0; j<aa_len; j++) {
                if (aa[j] != null_link) {
                    if (! first) fputc(' ', file);
                    first = 0;
                    if (reverse) fprintf(file, "%d-%d", (int)j, (int)aa[j]);
                    else fprintf(file, "%d-%d", (int)aa[j], (int)j);
                }
            }
        }
        fputc('\n', file);
    }

    Py_INCREF(Py_None);
    return Py_None;
}


// Initialize the parameters for one individual sampler.
static PyObject *py_gibbs_ibm_initialize(PyObject *self, PyObject *args) {
    PyObject *eee, *fff, *aaa, *counts_idx_arrays;
    PyArrayObject *counts_sum_array, *counts_array;
    PyArrayObject *jump_counts_array, *fert_counts_array;
    PyArrayObject *seed_array;
    unsigned long e_voc_size, f_voc_size;
    double lexical_alpha, null_alpha;
    int randomize;

    if(!PyArg_ParseTuple(args, "OOOOOOOOkkddOp",
                         &eee, &fff, &aaa, &counts_idx_arrays,
                         &counts_array, &counts_sum_array,
                         &jump_counts_array, &fert_counts_array,
                         &e_voc_size, &f_voc_size,
                         &lexical_alpha, &null_alpha,
                         &seed_array, &randomize))
        return NULL;
        
    PRNG_SEED_t *seed = (PRNG_SEED_t*) PyArray_GETPTR1(seed_array, 0);
    COUNT_t *counts = (COUNT_t*) PyArray_GETPTR1(counts_array, 0);
    COUNT_t *counts_sum = (COUNT_t*) PyArray_GETPTR1(counts_sum_array, 0);
    COUNT_t *jump_counts = ((PyObject*)jump_counts_array == Py_None)
        ? NULL
        : PyArray_GETPTR1(jump_counts_array, 0);
    COUNT_t *fert_counts = ((PyObject*)fert_counts_array == Py_None)
        ? NULL
        : PyArray_GETPTR1(fert_counts_array, 0);

    const size_t n_sents = PyTuple_Size(eee);
    const size_t counts_size = PyArray_DIM(counts_array, 0);

    for (size_t i=0; i<f_voc_size; i++)
        counts[i] = null_alpha;
    for (size_t i=f_voc_size; i<counts_size; i++)
        counts[i] = lexical_alpha;

    counts_sum[0] = null_alpha*(COUNT_t)f_voc_size;
    for (size_t i=1; i<e_voc_size; i++)
        counts_sum[i] = lexical_alpha*(COUNT_t)f_voc_size;

    for (size_t sent=0; sent<n_sents; sent++) {
        PyArrayObject *ee_array = (PyArrayObject*) PyTuple_GET_ITEM(eee, sent);
        PyArrayObject *ff_array = (PyArrayObject*) PyTuple_GET_ITEM(fff, sent);
        const size_t ee_len = (size_t) PyArray_DIM(ee_array, 0);
        const size_t ff_len = (size_t) PyArray_DIM(ff_array, 0);

        if (ee_len == 0 || ff_len == 0) continue;

        PyArrayObject *aa_array = (PyArrayObject*) PyTuple_GET_ITEM(aaa, sent);
        PyArrayObject *counts_idx_array =
            (PyArrayObject*) PyTuple_GET_ITEM(counts_idx_arrays, sent);

        const TOKEN_t *ee = (const TOKEN_t*) PyArray_GETPTR1(ee_array, 0);
        const TOKEN_t *ff = (const TOKEN_t*) PyArray_GETPTR1(ff_array, 0);
        LINK_t *aa = (LINK_t*) PyArray_GETPTR1(aa_array, 0);
        const INDEX_t *counts_idx = (const INDEX_t*) PyArray_GETPTR1(
            counts_idx_array, 0);

        int aa_jm1 = -1;
        if (randomize) {
            for (size_t j=0; j<ff_len; j++) {
                if (prng_next_count(seed) < 0.1) {
                    aa[j] = null_link;
                    counts[ff[j]] += (COUNT_t)1.0;
                    counts_sum[0] += (COUNT_t)1.0;
                } else {
                    const size_t i = prng_next_int(seed, ee_len);
                    aa[j] = i;
                    counts[counts_idx[i]] += (COUNT_t)1.0;
                    counts_sum[ee[i]] += (COUNT_t)1.0;
                    if (jump_counts != NULL) {
                        const size_t jump = get_jump_index(aa_jm1, i, ee_len);
                        aa_jm1 = i;
                        jump_counts[jump] += (COUNT_t)1.0;
                        jump_counts[JUMP_SUM] += (COUNT_t)1.0;
                    }
                }
                counts_idx += ee_len;
            }
        } else {
            for (size_t j=0; j<ff_len; j++) {
                if (aa[j] == null_link) {
                    counts[ff[j]] += (COUNT_t)1.0;
                    counts_sum[0] += (COUNT_t)1.0;
                } else {
                    const size_t i = (size_t)aa[j];
                    counts[counts_idx[i]] += (COUNT_t)1.0;
                    counts_sum[ee[i]] += (COUNT_t)1.0;
                    if (jump_counts != NULL) {
                        const size_t jump = get_jump_index(aa_jm1, i, ee_len);
                        aa_jm1 = i;
                        jump_counts[jump] += (COUNT_t)1.0;
                        jump_counts[JUMP_SUM] += (COUNT_t)1.0;
                    }
                }
                counts_idx += ee_len;
            }
        }
        if (fert_counts != NULL) {
            int fert[ee_len];
            for (size_t i=0; i<ee_len; i++)
                fert[i] = 0;
            for (size_t j=0; j<ff_len; j++)
                if (aa[j] != null_link) fert[aa[j]]++;
            for (size_t i=0; i<ee_len; i++)
                fert_counts[get_fert_index(ee[i], fert[i])] += (COUNT_t)1.0;
        }
        if (jump_counts != NULL && aa_jm1 >= 0) {
            jump_counts[get_jump_index(aa_jm1, ee_len, ee_len)] += (COUNT_t)1.0;
            jump_counts[JUMP_SUM] += (COUNT_t)1.0;
        }
    }

#if CACHE_RECIPROCAL
    for (size_t e=0; e<e_voc_size; e++)
        counts_sum[e] = (COUNT_t)1.0 / counts_sum[e];
#endif

    Py_INCREF(Py_None);
    return Py_None;
}


// Initialize the parameters for multiple individual samplers.
static PyObject *py_gibbs_ibm_initialize_parallel(
        PyObject *self, PyObject *args) {

    PyObject *eee, *fff, *counts_idx_arrays, *params;
    PyArrayObject *seed_array;
    unsigned long e_voc_size, f_voc_size;
    double lexical_alpha, null_alpha;
    int randomize;

    if(!PyArg_ParseTuple(args, "OOOOkkddOp",
                         &params, &eee, &fff, &counts_idx_arrays,
                         &e_voc_size, &f_voc_size,
                         &lexical_alpha, &null_alpha,
                         &seed_array, &randomize))
        return NULL;

    const size_t n_sents = PyTuple_Size(eee);
    const size_t n_samplers = PyTuple_Size(params);

    PRNG_SEED_t *seed = (PRNG_SEED_t*) PyArray_GETPTR1(seed_array, 0);
    PRNG_SEED_t local_seeds[n_samplers];
    for (size_t sampler=0; sampler<n_samplers; sampler++)
        local_seeds[sampler] = prng_next(*seed + 1 + sampler);
    *seed = prng_next(*seed);

#pragma omp parallel for
    for (size_t sampler=0; sampler<n_samplers; sampler++) {
        PRNG_SEED_t seed_cache = local_seeds[sampler];

        PyObject *sampler_params = PyTuple_GET_ITEM(params, sampler);

        PyObject *aaa = PyTuple_GET_ITEM(sampler_params, 0);
        PyArrayObject *counts_array =
            (PyArrayObject*) PyTuple_GET_ITEM(sampler_params, 1);
        PyArrayObject *counts_sum_array =
            (PyArrayObject*) PyTuple_GET_ITEM(sampler_params, 2);
        PyArrayObject *jump_counts_array =
            (PyArrayObject*) PyTuple_GET_ITEM(sampler_params, 3);
        PyArrayObject *fert_counts_array =
            (PyArrayObject*) PyTuple_GET_ITEM(sampler_params, 4);

        COUNT_t *counts = (COUNT_t*) PyArray_GETPTR1(counts_array, 0);
        COUNT_t *counts_sum = (COUNT_t*) PyArray_GETPTR1(counts_sum_array, 0);
        COUNT_t *jump_counts = ((PyObject*)jump_counts_array == Py_None)
            ? NULL
            : PyArray_GETPTR1(jump_counts_array, 0);
        COUNT_t *fert_counts = ((PyObject*)fert_counts_array == Py_None)
            ? NULL
            : PyArray_GETPTR1(fert_counts_array, 0);

        const size_t counts_size = PyArray_DIM(counts_array, 0);

        for (size_t i=0; i<f_voc_size; i++)
            counts[i] = null_alpha;
        for (size_t i=f_voc_size; i<counts_size; i++)
            counts[i] = lexical_alpha;

        counts_sum[0] = null_alpha*(COUNT_t)f_voc_size;
        for (size_t i=1; i<e_voc_size; i++)
            counts_sum[i] = lexical_alpha*(COUNT_t)f_voc_size;

        for (size_t sent=0; sent<n_sents; sent++) {
            PyArrayObject *ee_array =
                (PyArrayObject*) PyTuple_GET_ITEM(eee, sent);
            PyArrayObject *ff_array =
                (PyArrayObject*) PyTuple_GET_ITEM(fff, sent);
            const size_t ee_len = (size_t) PyArray_DIM(ee_array, 0);
            const size_t ff_len = (size_t) PyArray_DIM(ff_array, 0);

            if (ee_len == 0 || ff_len == 0) continue;

            PyArrayObject *aa_array =
                (PyArrayObject*) PyTuple_GET_ITEM(aaa, sent);
            PyArrayObject *counts_idx_array =
                (PyArrayObject*) PyTuple_GET_ITEM(counts_idx_arrays, sent);

            const TOKEN_t *ee = (const TOKEN_t*) PyArray_GETPTR1(ee_array, 0);
            const TOKEN_t *ff = (const TOKEN_t*) PyArray_GETPTR1(ff_array, 0);
            LINK_t *aa = (LINK_t*) PyArray_GETPTR1(aa_array, 0);
            const INDEX_t *counts_idx = (const INDEX_t*) PyArray_GETPTR1(
                counts_idx_array, 0);

            int aa_jm1 = -1;
            if (randomize) {
                for (size_t j=0; j<ff_len; j++) {
                    if (prng_next_count(&seed_cache) < 0.1) {
                        aa[j] = null_link;
                        counts[ff[j]] += (COUNT_t)1.0;
                        counts_sum[0] += (COUNT_t)1.0;
                    } else {
                        const size_t i = prng_next_int(&seed_cache, ee_len);
                        aa[j] = i;
                        counts[counts_idx[i]] += (COUNT_t)1.0;
                        counts_sum[ee[i]] += (COUNT_t)1.0;
                        if (jump_counts != NULL) {
                            const size_t jump =
                                get_jump_index(aa_jm1, i, ee_len);
                            aa_jm1 = i;
                            jump_counts[jump] += (COUNT_t)1.0;
                            jump_counts[JUMP_SUM] += (COUNT_t)1.0;
                        }
                    }
                    counts_idx += ee_len;
                }
            } else {
                for (size_t j=0; j<ff_len; j++) {
                    if (aa[j] == null_link) {
                        counts[ff[j]] += (COUNT_t)1.0;
                        counts_sum[0] += (COUNT_t)1.0;
                    } else {
                        const size_t i = (size_t)aa[j];
                        counts[counts_idx[i]] += (COUNT_t)1.0;
                        counts_sum[ee[i]] += (COUNT_t)1.0;
                        if (jump_counts != NULL) {
                            const size_t jump =
                                get_jump_index(aa_jm1, i, ee_len);
                            aa_jm1 = i;
                            jump_counts[jump] += (COUNT_t)1.0;
                            jump_counts[JUMP_SUM] += (COUNT_t)1.0;
                        }
                    }
                    counts_idx += ee_len;
                }
            }
            if (fert_counts != NULL) {
                int fert[ee_len];
                for (size_t i=0; i<ee_len; i++)
                    fert[i] = 0;
                for (size_t j=0; j<ff_len; j++)
                    if (aa[j] != null_link) fert[aa[j]]++;
                for (size_t i=0; i<ee_len; i++)
                    fert_counts[get_fert_index(ee[i], fert[i])] +=
                        (COUNT_t)1.0;
            }
            if (jump_counts != NULL && aa_jm1 >= 0) {
                jump_counts[get_jump_index(aa_jm1, ee_len, ee_len)] +=
                    (COUNT_t)1.0;
                jump_counts[JUMP_SUM] += (COUNT_t)1.0;
            }
        }

#if CACHE_RECIPROCAL
        for (size_t e=0; e<e_voc_size; e++)
            counts_sum[e] = (COUNT_t)1.0 / counts_sum[e];
#endif

        local_seeds[sampler] = seed_cache;
    }


    Py_INCREF(Py_None);
    return Py_None;
}



static PyMethodDef gibbsMethods[] = {
    {"ibm_sample_parallel", py_gibbs_ibm_sample_parallel, METH_VARARGS,
     "Sample from the model using multiple samplers in parallel\n\n"
     "n_iterations -- integer number of iterations per sampler\n"
     "model -- integer model (1 = IBM1, 2 = HMM, 3 = HMM+F\n"
     "eee -- tuple, source language sentences\n"
     "fff -- tuple, target language sentences\n"
     "dists -- buffer to add sampling distributions to, or None to skip\n"
     "counts_idx -- indexes into lexical counts array, during sampling one\n"
     "              simply advances one step per hypothesis in this array,\n"
     "              i.e. the length of the source sentence for each sample\n"
     "              counts_idx is a tuple with one such index vector per\n"
     "              sentence pair\n"
     "params -- n-tuple of 5-tuples, the latter are parameter vectors for\n"
     "          each of the n independent samplers (for details, please see\n"
     "          Aligner.create_sampler() in cyalign.pyx)\n"
     "null_prior -- prior probability of NULL alignments\n"
     "seed -- PRNG state\n"
    },
    {"ibm_create", py_gibbs_ibm_create, METH_VARARGS,
     "Initialize the lexical counts index\n\n"
     "eee -- tuple, source language sentences\n"
     "fff -- tuple, target language sentences\n"
     "counts_idx -- indexes into lexical counts array, to be initialized\n"
     "e_voc_size -- int, source of source vocabulary\n"
     "f_voc_size -- int, source of target vocabulary\n"
    },
    {"ibm_initialize", py_gibbs_ibm_initialize, METH_VARARGS,
     "Initialize parameters for a specific sampler"},
    {"ibm_initialize_parallel", py_gibbs_ibm_initialize_parallel, METH_VARARGS,
     "Initialize parameters for a number of samplers in parallel\n\n"
     "params -- tuple of 5-tuples containing the parameters\n"
     "eee -- tuple, source language sentences\n"
     "fff -- tuple, target language sentences\n"
     "counts_idx -- indexes into lexical counts array, to be initialized\n"
     "e_voc_size -- int, source of source vocabulary\n"
     "f_voc_size -- int, source of target vocabulary\n"
     "lexical_alpha -- Dirichlet parameter for lexical distributions\n"
     "null_alpha -- Dirichlet parameter for NULL lexical distribution\n"
     "seed -- PRNG state\n"
     "randomize -- if True, the alignments will be randomized\n"
    },
    {"ibm_discretize", py_gibbs_ibm_discretize, METH_VARARGS,
     "Discretize alignment distributions from a sampler\n\n"
     "dists -- sampling distributions\n"
     "aaa -- alignment variables to be written\n"
    },
    {"ibm_print", py_gibbs_ibm_print, METH_VARARGS,
     "Print discretized alignments\n\n"
     "aaa -- alignment variables\n"
     "reverse -- if True, reverse the source/target indexes\n"
     "fd -- file descriptor (integer) to write to\n"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef gibbsModule = {
    PyModuleDef_HEAD_INIT,
    "gibbs",
    NULL,
    -1,
    gibbsMethods
};

PyMODINIT_FUNC
PyInit_gibbs(void) {
    import_array();
    return PyModule_Create(&gibbsModule);
}

