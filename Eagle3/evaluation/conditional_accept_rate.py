import json
import argparse
from copy import deepcopy

from prettytable import PrettyTable

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
args = parser.parse_args()

f = open(args.input_file, 'r')
lines = f.readlines()
print('num of samples:', len(lines))

avg_accept_length = 0
avg_con_acc_ratio = {}

for line in lines:
    data = json.loads(line)
    avg_accept_length += sum(data['choices'][0]['accept_length']) / len(data['choices'][0]['accept_length']) + 1

    for acl in data['choices'][0]['accept_length']:
        if acl not in avg_con_acc_ratio.keys():
            avg_con_acc_ratio[acl] = 1
        else:
            avg_con_acc_ratio[acl] += 1

avg_accept_length /= len(lines)

# compute conditional accept ratio

cum_counter = []
for acl in range(len(avg_con_acc_ratio.keys())):
    cum_counter.append(avg_con_acc_ratio[acl])

rep_counter = deepcopy(cum_counter)
max_acl = len(cum_counter) - 1
while max_acl >= 0:
    for i in range(max_acl):
        cum_counter[i] += rep_counter[max_acl]
    max_acl -= 1

print(f"average acceptance length = {avg_accept_length:.5f}")
print(f"rep_counter = {rep_counter}")
print(f"cum_counter = {cum_counter}")

table = PrettyTable()
table.border = False
table.field_names = ['Position', 'Acceptance Rate']
for i in range(len(cum_counter) - 1):
    rate = cum_counter[i + 1] / cum_counter[i]
    table.add_row([f"{i + 1}", f"{rate:.5f}"])
print(f"\n{table}\n")
