import argparse
import json

import numpy as np
from prettytable import PrettyTable
from transformers import AutoTokenizer


def acceptance_length():
    f = open(args.LD_json, 'r')
    lines = f.readlines()
    print('num of samples:', len(lines))

    avg_accept_length = 0

    for line in lines:
        data = json.loads(line)
        avg_accept_length += sum(data['choices'][0]['accept_length']) / len(data['choices'][0]['accept_length']) + 1

    avg_accept_length /= len(lines)
    return round(avg_accept_length, 5)


def speedup_ratio():
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    jsonl_file = args.LD_json
    jsonl_file_base = args.baseline_json

    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    speeds = []
    for datapoint in data:
        qid = datapoint["question_id"]
        answer = datapoint["choices"][0]['turns']
        tokens = sum(datapoint["choices"][0]['new_tokens'])
        times = sum(datapoint["choices"][0]['wall_time'])
        speeds.append(tokens / times)

    data = []
    with open(jsonl_file_base, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    total_time = 0
    total_token = 0
    speeds0 = []
    for datapoint in data:
        qid = datapoint["question_id"]
        answer = datapoint["choices"][0]['turns']
        tokens = 0
        for i in answer:
            tokens += (len(tokenizer(i).input_ids) - 1)
        times = sum(datapoint["choices"][0]['wall_time'])
        speeds0.append(tokens / times)
        total_time += times
        total_token += tokens

    speedup_ratio = np.array(speeds).mean() / np.array(speeds0).mean()
    return round(speedup_ratio, 5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--baseline_json", type=str)
    parser.add_argument("--LD_json", type=str)
    args = parser.parse_args()

    benchmark = args.LD_json.split('/')[0]
    temperature = args.LD_json.split('-temperature-')[1][0]
    table = PrettyTable()
    table.field_names = ['Temperature', 'Benchmark', 'Speedup Ratio', 'Acceptance Length']
    # table.add_row([temperature, benchmark, speedup_ratio(), acceptance_length()])
    table.add_row([temperature, benchmark, '-', acceptance_length()])
    print(table)
