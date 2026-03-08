import argparse
import os
from concurrent.futures import ThreadPoolExecutor

from loguru import logger


def split_range(start, end, n, over=False):
    length = end - start + 1  # Include the end
    base_interval = length // n
    additional = length % n  # Get the remainder of the division
    intervals = []
    previous = start

    for i in range(n):
        current_interval = base_interval + (1 if i < additional else 0)
        if over:
            intervals.append((previous, previous + current_interval))
        else:
            intervals.append(
                (previous, previous + current_interval - 1)
            )  # '-1' because the end is inclusive
        previous += current_interval

    return intervals


def main():
    parser = argparse.ArgumentParser()

    # entrance
    parser.add_argument('--script', type=str)
    # path
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str)
    # data
    parser.add_argument('--dataset_name', type=str, default='ShareGPT')
    parser.add_argument('--num_rows', type=int, default=68000)
    parser.add_argument('--num_gpus', type=int, default=4)

    args = parser.parse_args()

    dataset_name = args.dataset_name
    num_rows = args.num_rows
    num_gpus = args.num_gpus

    outdir = f'{args.outdir}/{dataset_name}_{num_rows}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    intervals = split_range(0, num_rows - 1, num_gpus, over=True)

    commands = []
    for index in range(num_gpus):
        start, end = intervals[index]

        command = (
            f'CUDA_VISIBLE_DEVICES={index} python {args.script} '
            f'--start {start} '
            f'--end {end} '
            f'--index {index} '
            f'--outdir {outdir} '
            f'--data_path {args.data_path} '
            f'--model_path {args.model_path} '
            f'--dataset_name {dataset_name}'
        )
        commands.append(command)
    logger.info('\n'.join(commands))

    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        for command in commands:
            executor.submit(lambda cmd: os.system(cmd), command)


if __name__ == '__main__':
    main()
