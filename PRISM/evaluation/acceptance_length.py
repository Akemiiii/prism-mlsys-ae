import json
import argparse

from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
args = parser.parse_args()

f = open(args.input_file, 'r')
lines = f.readlines()
print('num of samples:', len(lines))

avg_accept_length = 0

for line in lines:
    data = json.loads(line)
    avg_accept_length += sum(data['choices'][0]['accept_length']) / len(data['choices'][0]['accept_length']) + 1

avg_accept_length /= len(lines)

logger.info(f'[input_file = {args.input_file}, acceptance_length = {avg_accept_length:.5f}]')
