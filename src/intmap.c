// Simple uint64_t -> uint64_t hash map implementation.
//
// Keys and values may not be equal to INTMAP_EMPTY_KEY or INTMAP_EMPTY_VALUE,
// respectively.
//
// Deletion is not implemented, and the main use is intmap_setdefault()

#include <stdint.h>


// If these are changed, make sure to modify the memset() calls below
#define INTMAP_EMPTY_KEY    0xffffffffffffffffULL
#define INTMAP_EMPTY_VALUE  0xffffffffffffffffULL

// Aim at a maximum capacity of 1/INTMAP_CAPACITY (should be at least 2)
#define INTMAP_CAPACITY     2

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

// xxHash64's finalization step
static inline uint64_t hash(uint64_t x) {
    x = (x ^ (x >> 33)) * 14029467366897019727ULL;
    x = (x ^ (x >> 29)) * 1609587929392839161ULL;
    return x ^ (x >> 32);
}

typedef struct {
    size_t size;
    size_t n;
    uint64_t *buf;
} intmap;

static int intmap_set(intmap *im, uint64_t k, uint64_t v);


static int intmap_create(intmap *im, size_t size) {
    if (size & (size-1)) return 1;
    im->n = 0;
    im->size = size;
    im->buf = malloc(im->size * sizeof(uint64_t));
    if (im->buf == NULL) return 1;
    memset(im->buf, 0xff, im->size * sizeof(uint64_t));
    return 0;
}

static void intmap_free(intmap *im) {
    free(im->buf);
    im->buf = NULL;
}

static int intmap_expand(intmap *im) {
    const uint64_t *old_buf = im->buf;
    const size_t old_size = im->size;

    if (im->size == 0x8000000000000000ULL) return 1;
    im->size *= 2;
    im->n = 0;
    im->buf = malloc(im->size * sizeof(uint64_t));
    if (im->buf == NULL) return 1;
    memset(im->buf, 0xff, im->size * sizeof(uint64_t));
    for (size_t i=0; i<old_size; i+=2) {
        if (old_buf[i] != INTMAP_EMPTY_KEY) {
            intmap_set(im, old_buf[i], old_buf[i+1]);
        }
    }
    return 0;
}

static inline size_t intmap_get_slot(const intmap *im, uint64_t k) {
    const size_t mask = im->size - 1;
    size_t i = ((size_t) hash(k) * 2) & mask;

    while (1) {
        if (im->buf[i] == k) return i;
        if (im->buf[i] == INTMAP_EMPTY_KEY) return i;
        i = (i + 2) & mask;
    }
}

//static uint64_t intmap_get(const intmap *im, uint64_t k) {
//    return im->buf[intmap_get_slot(im, k)+1];
//}

static int intmap_set(intmap *im, uint64_t k, uint64_t v) {
    if (im->n*INTMAP_CAPACITY*2 > im->size)
            if (intmap_expand(im) < 0) return -1;
    const size_t i = intmap_get_slot(im, k);
    if (im->buf[i] == INTMAP_EMPTY_KEY) {
        im->n++;
        im->buf[i] = k;
        im->buf[i+1] = v;
        return 1;
    }
    im->buf[i+1] = v;
    return 0;
}

static uint64_t intmap_setdefault(intmap *im, uint64_t k, uint64_t v) {
    const size_t i = intmap_get_slot(im, k);
    if (im->buf[i] == INTMAP_EMPTY_KEY) {
        im->n++;
        im->buf[i] = k;
        im->buf[i+1] = v;
        if (im->n*INTMAP_CAPACITY*2 > im->size)
                if (intmap_expand(im) < 0) return -1;
        return v;
    }
    return im->buf[i+1];
}

