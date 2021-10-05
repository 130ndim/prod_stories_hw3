import argparse
import os.path as osp

import numpy as np
from scipy.stats import rankdata


def monotone_conjunction_test(data: np.ndarray) -> tuple[int, int, float]:
    if data.ndim != 2 and data.shape[1] != 2:
        raise ValueError(
            'Expected a set of 2D points. Got input of shape %s.' % str(data.shape)
        )
    n = data.shape[0]
    if n < 9:
        raise ValueError(
            'Monotone conjuction test expects at least 9 points in a set! '
            'Got %s points' % n
        )
    ys = data[np.argsort(data[:, 0]), 1]

    print(ys)
    ranks = rankdata(ys)
    ranks = -ranks + ranks.max() + 1  # type: ignore

    p = int(round(n / 3))
    r1 = ranks[:p].sum()
    r2 = ranks[-p:].sum()

    diff = r1 - r2
    standard_error = (n + 0.5) * (p / 6) ** 0.5
    conjunction = diff / (p * (n - p))

    diff = int(round(diff))
    standard_error = int(round(standard_error))
    conjunction = round(conjunction, 2)

    return diff, standard_error, conjunction


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='./in.txt', type=str, help='Input file path')
    ap.add_argument('--output', default='./out.txt', type=str, help='Output file path')
    args = ap.parse_args()

    if not osp.exists(args.input) or osp.isdir(args.input):
        raise FileNotFoundError('No such file: %s' % args.input)

    points = []
    with open(args.input, 'r') as f:
        for i, line in enumerate(f, start=1):
            r = line.split()
            if not all(map(str.isdigit, r)) or len(r) != 2:
                raise ValueError('Wrong value in line %s!' % i)
            points.append(list(map(int, r)))

    points = np.array(points)

    out = monotone_conjunction_test(points)
    with open(args.output, 'w') as f:
        f.write(' '.join(map(str, out)))
