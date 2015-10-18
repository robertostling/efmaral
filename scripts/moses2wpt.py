#!/usr/bin/env python3

# Script to convert Moses-style alignments to WPT-style.
#
# Usage: ./moses2wpt.py file.moses >file.wa

import sys

if len(sys.argv) != 2:
    print('Usage: %s file.moses >file.wa' % __file__, file=sys.stderr)
    sys.exit()

with open(sys.argv[1]) as f:
    for i,line in enumerate(f):
        for pair in line.split():
            j,a = pair.split('-')
            print('%d %s %s' % (i+1, j, a))

