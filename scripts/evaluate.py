#!/usr/bin/env python3

# Script to evaluate efmaral or fast_align using WPT shared task data sets.
#
#   time python3 wpteval.py efmaral test.eng.hin.wa test.eng test.hin \
#       training.eng training.hin
#
# or, to use fast_align:
#
#   time python3 wpteval.py fast_align test.eng.hin.wa test.eng test.hin \
#       training.eng training.hin
#
# atools (from the fast_align package) must be installed and in $PATH

import re, sys, subprocess, os
from multiprocessing import Pool

RE_NUMBERED = re.compile(r'<s snum=(\d+)>(.*?)</s>\s*$')

def wpteval(align, train_filenames, test_filename, gold_wa):
    test_numbers = []
    text1_filename = 'text1.txt'
    text2_filename = 'text2.txt'
    align_filename = 'text1-text2.moses'
    align_wa = 'text1-text2.wa'

    with open(text1_filename, 'w', encoding='utf-8') as outf1, \
         open(text2_filename, 'w', encoding='utf-8') as outf2:
        with open(test_filename[0], 'r', encoding='utf-8') as f:
            for i,line in enumerate(f):
                m = RE_NUMBERED.match(line)
                assert m, 'Test data file %s not numbered properly!' % \
                          test_filename[0]
                test_numbers.append(m.group(1))
                print(m.group(2).strip(), file=outf1)
        with open(test_filename[1], 'r', encoding='utf-8') as f:
            for i,line in enumerate(f):
                m = RE_NUMBERED.match(line)
                assert m, 'Test data file %s not numbered properly!' % \
                          test_filename[1]
                assert test_numbers[i] == m.group(1)
                print(m.group(2).strip(), file=outf2)

        for filename1,filename2 in train_filenames:
            with open(filename1, 'r', encoding='utf-8') as f1, \
                 open(filename2, 'r', encoding='utf-8') as f2:
                while True:
                    line1 = f1.readline()
                    line2 = f2.readline()
                    assert (not line1) == (not line2), \
                           'Number of lines differs between %s and %s!' % (
                           filename1, filename2)
                    if (not line1) or (not line2): break
                    line1 = line1.strip()
                    line2 = line2.strip()
                    if line1 and line2:
                        print(line1, file=outf1)
                        print(line2, file=outf2)

    align(text1_filename, text2_filename, align_filename)

    with open(align_filename, 'r') as f, \
         open(align_wa, 'w') as outf:
        for lineno in test_numbers:
            for i,j in map(lambda s: s.split('-'), f.readline().split()):
                print('%s %d %d' % (lineno, int(i)+1, int(j)+1), file=outf)

    subprocess.call(['perl', '3rdparty/wa_check_align.pl', align_wa])
    subprocess.call(['perl', '3rdparty/wa_eval_align.pl', gold_wa, align_wa])

    os.remove(text1_filename)
    os.remove(text2_filename)
    os.remove(align_filename)
    os.remove(align_wa)


def fastalign(args):
    in_filename, out_filename, reverse = args
    with open(out_filename, 'w') as outf:
        subprocess.call(
            ['fast_align', '-i', in_filename, '-d', '-o', '-v']
            if reverse else 
            ['fast_align', '-i', in_filename, '-d', '-o', '-v', '-r'],
            stdout=outf)


def main():
    def align_efmaral(text1, text2, output):
        subprocess.call(['scripts/align_symmetrize.sh', text1, text2, output])

    def align_fastalign(text1, text2, output):
        tmp_filename = 'fastalign.txt'
        fwd_filename = 'fastalign.fwd'
        back_filename = 'fastalign.back'
        with open(tmp_filename, 'w') as outf:
            subprocess.call(['scripts/wpt2fastalign.py', text1, text2],
                            stdout=outf)
    
        with Pool(2) as p:
            r = p.map(fastalign,
                      [(tmp_filename, fwd_filename, False),
                       (tmp_filename, back_filename, True)])

        os.remove(tmp_filename)
        with open(output, 'w') as outf:
            subprocess.call(['atools', '-i', fwd_filename, '-j', back_filename,
                             '-c', 'grow-diag-final-and'], stdout=outf)
        os.remove(fwd_filename)
        os.remove(back_filename)

    aligner = align_efmaral if sys.argv[1] == 'efmaral' else align_fastalign
    wpteval(aligner,
            zip(sys.argv[5].split(','), sys.argv[6].split(',')),
            (sys.argv[3], sys.argv[4]),
            sys.argv[2])

if __name__ == '__main__': main()

