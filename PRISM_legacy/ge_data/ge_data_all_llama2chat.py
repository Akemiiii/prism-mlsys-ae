import argparse
import os
import re

import torch
from datasets import concatenate_datasets, load_dataset
from fastchat.model.model_adapter import get_conversation_template
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
        "conv_key": ["conversations"],
        "split_key": ["train"],
    },
    "UltraChat": {
        "load_type": "datasets",
        "conv_key": ["messages"],
        "split_key": ["train_sft", "train_gen"],
    },
    "OpenThoughts2": {
        "load_type": "datasets",
        "conv_key": ["conversations"],
        "split_key": ["train"],
    },
    "Baize": {
        "load_type": "datasets",
        "conv_key": ["chat_sample"],
        "split_key": ["train"],
    },
    "evol_instruct": {
        "load_type": "datasets",
        "conv_key": ["instruction", "output"],
        "split_key": ["train"],
    },
    "lima": {
        "load_type": "json",
        "conv_key": ["conversations"],
        "split_key": ["train"],
    },
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


# for different datasets, we just need to change the way we fill the conv template
def fill_conv_template(dataset_name, conv, source):
    # 1. Get the roles
    if dataset_name == "ShareGPT":
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    elif dataset_name == "UltraChat":
        roles = {"user": conv.roles[0], "assistant": conv.roles[1]}
    elif dataset_name == "OpenThoughts2":
        roles = {"user": conv.roles[0], "assistant": conv.roles[1]}
    elif dataset_name == "Baize":
        roles = {"Human": conv.roles[0], "AI": conv.roles[1]}
        # in baize data, input source is a string
    elif dataset_name == "evol_instruct":
        # two keys for evol_instruct
        # don't need a role
        pass
    elif dataset_name == "lima":
        # don't need a role
        pass

    # 2. customize fill
    if dataset_name == "ShareGPT":
        source = source[0]
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            if sentence["from"] == "gpt":
                sentence["value"] = " " + sentence["value"]
            conv.append_message(role, sentence["value"])

    elif dataset_name == "UltraChat":
        source = source[0]
        if roles[source[0]["role"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["role"]]
            assert role == conv.roles[j % 2]
            if sentence["role"] == "assistant":
                sentence["content"] = " " + sentence["content"]
            conv.append_message(role, sentence["content"])

    elif dataset_name == "OpenThoughts2":
        source = source[0]
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            if sentence["from"] == "assistant":
                # also need delete the thinking tokens
                sentence["value"] = re.sub(
                    r"<think>.*?</think>", "", sentence["value"], flags=re.DOTALL
                ).strip()
                sentence["value"] = " " + sentence["value"]
            conv.append_message(role, sentence["value"])

    elif dataset_name == "Baize":
        source = source[0]
        # use regex to analyze the string
        pattern = re.compile(
            r"\[\|(Human|AI)\|\]\s*(.*?)(?=\s*\[\|(?:Human|AI)\|\]|$)", re.DOTALL
        )
        for j, match in enumerate(pattern.finditer(source)):
            role = match.group(1).strip()
            value = match.group(2).strip()

            if j == 0:
                offset = 0
                if roles[role] != conv.roles[0]:
                    offset = 1
                    print("triggered!")
                    continue

            assert roles[role] == conv.roles[(j + offset) % 2]
            if role == "AI":
                value = " " + value
            conv.append_message(role, value)

    elif dataset_name == "evol_instruct":
        # there is no such thing as a turn here
        conv.append_message(conv.roles[0], source[0])
        conv.append_message(conv.roles[1], " " + source[1])

    elif dataset_name == "lima":
        source = source[0]
        for j, sentence in enumerate(source):
            if j % 2 == 0:
                conv.append_message(conv.roles[0], sentence)
            else:
                conv.append_message(conv.roles[1], " " + sentence)

    return conv


def build_dataset_rank(tokenizer):
    info = dataset_info[args.dataset_name]
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
    # ds1 = ds.select(range(100,200))
    # dst=ds.select(range(200,300))
    # ds2=ds.select(range(300,len(ds)))
    original_columns1 = ds1.column_names
    # original_columns2 = ds2.column_names
    num_proc = 4

    def preprocess_function(examples):
        new_examples = {"conversation": [], "input_ids": [], "loss_mask": []}
        for i in range(len(examples[info["conv_key"][0]])):
            conv = get_conversation_template("llama-2-chat")
            sys_p = (
                "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  "
                "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
                "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
                "If you don't know the answer to a question, please don't share false information."
            )
            conv.system_message = sys_p
            source = [examples[key][i] for key in info["conv_key"]]
            """
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                if sentence["from"] == "gpt":
                    sentence["value"] = " " + sentence["value"]
                conv.append_message(role, sentence["value"])
            """
            conv = fill_conv_template(args.dataset_name, conv, source)
            conversation = conv.get_prompt()

            # if i==56:
            #     print(i)
            # if i==57:
            #     print(i)
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=2048,
                truncation=True,
            ).input_ids[0]
            loss_mask = torch.ones_like(input_ids)
            # print(i)

            sep = conv.sep + conv.roles[1] + " "

            total_len = int(input_ids.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep2)
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
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                # if i != 0 and not tokenizer.legacy:
                #     # The legacy and non-legacy modes handle special tokens differently
                #     instruction_len -= 1

                # Ignore the user instructions
                loss_mask[cur_len: cur_len + instruction_len] = 0
                cur_len += turn_len
                cur_len += 2

                if i != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    cur_len -= 1

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
        load_from_cache_file=False,
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
    outs_big = bigmodel(input_ids.cuda(), output_hidden_states=True)
    hidden_state_big = outs_big.hidden_states[-1]
    max_prob_tokens_big = torch.argmax(outs_big.logits, dim=-1)
    probs = torch.softmax(outs_big.logits, dim=-1)
    maxp = probs[0].max(dim=1).values
    td = {
        "input_ids": input_ids.cpu()[0],
        "hidden_state": hidden_state_big.cpu()[0],
        "loss_mask": data["loss_mask"].cpu()[0],
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
