import argparse
import os
import re

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=100)
parser.add_argument("--index", type=int, default=0)
parser.add_argument("--outdir", type=str, default="0")
parser.add_argument("--data_path", type=str, default="0")
parser.add_argument("--model_path", type=str, default="0")
parser.add_argument("--dataset_name", type=str, default="ShareGPT")
args = parser.parse_args()

bigname = args.model_path

dataset_info = {
    "ShareGPT": {
        "load_type": "json",
        "conv_key": "conversations",
        "split_key": ["train"],
        "role_key": "from",
        "content_key": "value"
    },
    "UltraChat": {
        "load_type": "datasets",
        "conv_key": "messages",
        "split_key": ["train_sft", "train_gen"],
        "role_key": "role",
        "content_key": "content"
    },
    "OpenThoughts2": {
        "load_type": "datasets",
        "conv_key": "conversations",
        "split_key": ["train"],
        "role_key": "from",
        "content_key": "value"
    }
}


def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break

    common_prefix = list1[:prefix_length]
    return common_prefix, prefix_length


def build_dataset_rank(tokenizer):
    dataset_name = args.dataset_name
    info = dataset_info[dataset_name]

    if info["load_type"] == "json":
        ds = load_dataset("json", data_files=args.data_path)
    elif info["load_type"] == "datasets":
        ds = load_dataset(args.data_path)

    splits = []
    for key in info["split_key"]:
        if key in ds:
            splits.append(ds[key])
        else:
            print(
                f"Warning: Split key '{key}' not found in loaded dataset for {args.dataset_name}."
            )
    if not splits:
        print("No data splits found to merge.")
        merged_dataset = None  # Or handle as an error, or return an empty dataset
    else:
        # Merge all dataset objects in the 'splits' list
        merged_dataset = concatenate_datasets(splits)
        print(f"Successfully merged the following splits: {info['split_key']}")
        print(f"Total examples in merged dataset: {len(merged_dataset)}")

    ds = merged_dataset

    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))
    original_columns1 = ds1.column_names
    num_proc = 4

    def preprocess_function(examples):
        new_examples = {"conversation": [], "input_ids": [], "loss_mask": []}
        for i in range(len(examples[info["conv_key"]])):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  "
                        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
                        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
                        "If you don't know the answer to a question, please don't share false information."
                    )
                }
            ]

            convroles = ["user", "assistant"]
            if dataset_name == "ShareGPT":
                roles = {"human": convroles[0], "gpt": convroles[1]}
            else:
                roles = {"user": convroles[0], "assistant": convroles[1]}

            source = examples[info["conv_key"]][i]
            if roles[source[0][info["role_key"]]] != "user":
                # Skip the first one if it is not from human
                source = source[1:]

            for j, sentence in enumerate(source):
                role = roles[sentence[info["role_key"]]]
                assert role == convroles[j % 2], f"{i}"
                content = sentence[info["content_key"]]

                if role == "assistant":
                    if dataset_name == "OpenThoughts2":
                        content = re.sub(
                            r"<think>.*?</think>", "", content, flags=re.DOTALL
                        ).strip()
                    content = " " + content

                messages.append(
                    {"role": role, "content": content}
                )

            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=2048,
                truncation=True,
                add_special_tokens=False
            ).input_ids[0]
            loss_mask = torch.ones_like(input_ids)
            # print(i)

            sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            total_len = len(input_ids)

            sep2 = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
            turns = conversation.split(sep2)

            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]

            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

                # Ignore the user instructions
                if i == 0:
                    loss_mask[cur_len: cur_len + instruction_len - 2] = 0
                else:
                    loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0
                cur_len += turn_len
                if i != 0:
                    cur_len += 3

            loss_mask[cur_len:] = 0

            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )

    # ds1 = ds1.filter(lambda x: len(x["input_ids"]) < 1024, batched=False)
    # ds1 = ds1.filter(lambda x: x['queryf'] not in gqs, batched=False)
    # ds1 = ds1.filter(lambda x: "Are there any tips in regards to teaching" in x['queryf'], batched=False)

    ds1.set_format(type="torch")
    # ds2.set_format(type="torch")
    # dst.set_format(type="torch")
    return ds1


bigtokenizer = AutoTokenizer.from_pretrained(bigname, use_fast=False)
ds = build_dataset_rank(bigtokenizer)
print(ds)
# quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#     )
# bigmodel = AutoModelForCausalLM.from_pretrained(bigname, load_in_4bit=True, device_map={"": 0}, )
# smallmodel = AutoModelForCausalLM.from_pretrained(smallname, load_in_4bit=True, device_map={"": 1}, )
bigmodel = AutoModelForCausalLM.from_pretrained(
    bigname, device_map="auto", torch_dtype=torch.float16
)
# bigmodel = AutoModelForCausalLM.from_pretrained(bigname,  device_map="auto",load_in_8bit=True)
bigmodel.eval()


@torch.no_grad()
def ge(data):
    input_ids = data["input_ids"]
    outputs = bigmodel(input_ids.cuda(), output_hidden_states=True)

    hidden_state = torch.cat([
        outputs.hidden_states[3], outputs.hidden_states[17], outputs.hidden_states[30]
    ], dim=-1)
    target = outputs.hidden_states[-1]

    td = {
        "input_ids": input_ids.cpu()[0],
        "hidden_state": hidden_state.cpu()[0],
        "target": target.cpu()[0],
        "loss_mask": data["loss_mask"].cpu()[0]
    }
    return td


outdir = f"{args.outdir}/{args.index}"
if not os.path.exists(outdir):
    os.makedirs(outdir)


def writedata(name, data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length = len(os.listdir(name))
    idx = current_length
    torch.save(data_point, f"{name}/data_{idx}.ckpt")


for id, data in enumerate(ds):
    if id % 100 == 0:
        print(id, end="\t")
    if id % 1000 == 0:
        print("")
    outdata = ge(data)
    writedata(outdir, outdata)
