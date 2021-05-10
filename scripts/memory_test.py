import tracemalloc

from tqdm import tqdm

import numpy as np

ARRAY_SHAPE = (2770, 2770)
N_ARRAYS = 360


def do_stuff():
    out_dict = dict()
    for i in tqdm(range(N_ARRAYS)):
        out_dict[i] = np.random.rand(*ARRAY_SHAPE)*0.001

    return out_dict


def main():
    tracemalloc.start()
    do_stuff()
    current, peak = tracemalloc.get_traced_memory()
    current_gb = current / 10 ** 9
    peak_gb = peak / 10 ** 9
    print(
        f"Current memory usage is %.2fGB; Peak was %.2fGB"
        % (current_gb, peak_gb)
    )
    tracemalloc.stop()


if __name__ == '__main__':
    main()
