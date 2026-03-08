import argparse
import random

from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=str, default='LD')
parser.add_argument('--name', type=str, default='HASS')
parser.add_argument('--basepath', type=str, default=None)
parser.add_argument('--configpath', type=str, default=None)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
parser.add_argument('--tmpdir', type=str, default=None)
parser.add_argument('--cpdir', type=str, default=None)
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--topk', type=int, default=10)
parser.add_argument('--topk_w', type=float, default=1.0)
parser.add_argument('--forward_num_total', type=int, default=3)
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--data_num', type=int, default=68000)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--use_adapter', action='store_true')
parser.add_argument('--detach', action='store_true')
parser.add_argument('--hass_path', type=str, default=None)
parser.add_argument('--p_w', type=float, default=0.1)
parser.add_argument('--v_w', type=float, default=1.0)
parser.add_argument('--max_len', type=int, default=2048)

args = parser.parse_args()

total_steps = int(args.data_num * 0.95 * (args.epoch + 1) / (args.bs * args.gradient_accumulation_steps))
warm_steps = total_steps // 100

train_config = {
    "data_num": args.data_num,
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": args.epoch,
    # Depending on your data and model size, the larger the model, the higher the sample efficiency. We recommend setting it between 20-40.
    "num_warmup_steps": warm_steps,
    "total_steps": total_steps,
    "p_w": args.p_w,
    "v_w": args.v_w,
    "topk_w": args.topk_w,
    "head_w": 0.1,
    "num_workers": 8,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": args.max_len,
    # During training, truncating the training sequences means that the larger the setting, the more training data is used, and the better the effect, but it also consumes more VRAM.
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 1
}

import json
import safetensors
from safetensors import safe_open

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1
import torch

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(42)
accelerator = Accelerator(
    mixed_precision='bf16',
    gradient_accumulation_steps=train_config["gradient_accumulation_steps"]
)

from LD.large_drafter import LargeDrafter

from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoConfig

if accelerator.is_main_process:
    import wandb

    wandb.init(entity='ciai-llm', project=args.project, name=args.name, mode='offline', config=train_config)

baseconfig = AutoConfig.from_pretrained(args.basepath)

head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)

try:
    with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(args.basepath, head_path),
                   framework="pt",
                   device="cpu") as f:
        tensor_slice = f.get_slice("lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
except:
    with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    weights = torch.load(os.path.join(args.basepath, head_path))
    tensor = weights["lm_head.weight"].float()

head.weight.data = tensor
head.eval()

for param in head.parameters():
    param.requires_grad = False


def list_files(
        path: str,
        index_file_name: str = 'index.txt',
        num: int = 800000
):
    datapath = []

    index_file_path = os.path.join(path, index_file_name)
    if os.path.exists(index_file_path):
        logger.info('data index file exists.')
        with open(index_file_path, mode='r', encoding='utf-8') as reader:
            for line in reader:
                file_path = line.strip()
                datapath.append(file_path)
    else:
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                datapath.append(file_path)

        random.seed(42)
        random.shuffle(datapath)
        with open(index_file_path, mode='w', encoding='utf-8') as writer:
            for file_path in datapath:
                writer.write(file_path + '\n')

    logger.info(f'there are {len(datapath)} samples, select first {num}.')
    return datapath[:num]


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # try:
        data = torch.load(self.data[index])
        new_data = {}

        hidden_state = data['hidden_state'][:train_config["max_len"]][None, :]
        target = data['target'][:train_config["max_len"]][None, :]
        input_ids = data['input_ids'][:train_config["max_len"]][None, :]
        loss_mask = data["loss_mask"][:train_config["max_len"]][None, :]

        length = hidden_state.shape[1]
        # length_q = data['query_ids'].shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = target[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0

        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state"] = hidden_state
        new_data["input_ids"] = input_ids_target

        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


def top_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def compute_loss(target, target_p, predict, loss_mask):
    out_head = head(predict)
    out_logp = nn.LogSoftmax(dim=2)(out_head)

    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum() + 1e-5)

    vloss = criterion(predict, target)
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum() + 1e-5)

    topk_mask = torch.topk(target_p, k=args.topk, dim=2).indices
    topk_loss = -torch.sum(torch.sum(loss_mask * plogp.gather(dim=2, index=topk_mask), 2)) / (loss_mask.sum() + 1e-5)

    return vloss, ploss, topk_loss, out_head


@torch.no_grad()
def getkacc(model, data, head, max_length=5):
    def generate(hidden_states, input_ids, head, max_length=4, use_cache=True):
        if use_cache:
            past_key_values = None
            unwarped_model = accelerator.unwrap_model(model)
            unwarped_model.reset_step()
            for i in range(max_length):
                if i < args.forward_num_total:
                    assert i == unwarped_model.current_step, i
                if past_key_values != None:
                    out_hidden, past_key_values, sample_hidden = model(last_hidden, input_ids=token,
                                                                       past_key_values=past_key_values,
                                                                       use_cache=True)
                else:
                    out_hidden, past_key_values, sample_hidden = model(hidden_states, input_ids=input_ids,
                                                                       use_cache=True)
                last_hidden = out_hidden[:, -1:]
                if not args.use_adapter:
                    last_headout = head(last_hidden)
                else:
                    last_headout = head(sample_hidden[:, -1:])
                token = torch.argmax(last_headout, dim=-1)
                input_ids = torch.cat((input_ids, token), dim=1)
        else:
            raise NotImplementedError

        return input_ids

    hidden_states = data["hidden_states"]
    hidden_states = model.module.fusion_layer(hidden_states)
    input_ids = data["input_ids"]
    loss_mask = data["loss_mask"]
    target = data["target"]

    total = [0 for _ in range(max_length)]
    correct = [0 for _ in range(max_length)]
    bs, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    target_headout = head(target)
    target_ids = target_headout.argmax(dim=2)

    for pre_len in range(1, seq_len):
        if loss_mask[:, pre_len].sum() == 0:
            continue
        pre_hidden_states = hidden_states[:, :pre_len]
        pre_input_ids = input_ids[:, :pre_len]
        outs = generate(pre_hidden_states, pre_input_ids, head, max_length=max_length)
        generate_ids = outs[:, pre_len:]
        for bid in range(bs):
            for k in range(max_length):
                if loss_mask[bid, pre_len + k] == 0:
                    break
                if pre_len + k >= seq_len:
                    break
                total[k] += 1
                if generate_ids[bid, k] == target_ids[bid, pre_len + k - 1]:
                    correct[k] += 1
                else:
                    for kk in range(k + 1, max_length):
                        total[kk] += 1
                    break

    acc = [correct[i] / total[i] for i in range(len(correct))]
    return acc


datapath = list_files(train_config["datapath"], num=train_config["data_num"])

traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]

traindataset = CustomDataset(traindatapath)
testdataset = CustomDataset(testdatapath)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True,
                          collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                          pin_memory=True)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                         collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)

if accelerator.is_main_process:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

with open(train_config["config_path"]) as f:
    config = json.load(f)
assert config.get("use_adapter", False) == args.use_adapter

model = LargeDrafter(config, load_emb=True, path=args.basepath, hass_path=args.hass_path)
if accelerator.is_main_process:
    logger.info(model)

if args.ckpt_path is not None:
    ea_model_path = args.ckpt_path
    load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
    if os.path.exists(load_model_path):
        ea_layer_state_dict = torch.load(load_model_path, map_location="cuda")
    else:
        load_model_path = os.path.join(ea_model_path, "model.safetensors")
        ea_layer_state_dict = safetensors.torch.load_file(load_model_path)
    model.load_state_dict(ea_layer_state_dict, strict=True)
    print(f"load model from {load_model_path}")

criterion = nn.SmoothL1Loss(reduction="none")
optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

if is_warmup:
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)

    model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader, scheduler
    )
else:
    model, head, optimizer, train_loader, test_loader = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader
    )

# work-around: DDP 不支持 嵌套地gradient checkpoint计算
model._set_static_graph()
unwarped_model = accelerator.unwrap_model(model)

for epoch in range(num_epochs + 1):
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    for batch_idx, data in enumerate(tqdm(train_loader)):

        if args.debug and batch_idx > 10:
            break

        with accelerator.accumulate(model):
            optimizer.zero_grad()
            hidden_states, input_ids, attention_mask, target, loss_mask = data["hidden_states"], data["input_ids"], \
                data["attention_mask"], data["target"], data["loss_mask"][..., None]
            loss = 0
            with torch.no_grad():
                target_head = head(target)
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()

            hidden_states = model.module.fusion_layer(hidden_states)
            q_hidden_states = None  ### q hidden states is used to store past step's hidden states
            unwarped_model.reset_step()
            for forward_idx in range(args.forward_num_total):  ### forward for multiple times
                assert forward_idx == unwarped_model.current_step

                predict, sample_hidden = model(hidden_states, input_ids, attention_mask,
                                               q_hidden_states=q_hidden_states)  ### for me. just enable the model to switch paratmers is enough

                if q_hidden_states is None:
                    q_hidden_states = torch.cat([hidden_states[:, :1, :], predict[:, :-1, :]], dim=1)[None, :, :, :]
                    ### see here, q_hidden states is built by using the first in the sequence and add the predict into the sequence(discard the last), then it's the same size as the input hidden_states
                    ### then an additional dimension is added
                else:
                    new_q_hidden_states = torch.cat([q_hidden_states[-1][:, :1, :], predict[:, :-1, :]], dim=1)[None, :,
                    :, :]
                    q_hidden_states = torch.cat([q_hidden_states, new_q_hidden_states], dim=0)
                    ### q_hidden_states always maintains the hidden states of different generation steps

                if args.detach:
                    q_hidden_states = q_hidden_states.detach()

                if not args.use_adapter:
                    vloss, ploss, topk_loss, out_head = compute_loss(target, target_p, predict, loss_mask)
                else:
                    vloss, ploss, topk_loss, out_head = compute_loss(target, target_p, sample_hidden, loss_mask)
                total_loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss + train_config[
                    "topk_w"] * topk_loss
                loss += total_loss

            accelerator.backward(loss)

            accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            optimizer.zero_grad()
            loss /= args.forward_num_total
            if is_warmup:
                scheduler.step()

        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
        if accelerator.is_main_process and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                       "train/ploss": ploss.item(), "train/topkloss": topk_loss.item(), "train/loss": loss.item(),
                       "train/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct
            wandb.log(logdict)
            # for id,i in enumerate(top_3acc):
            #     wandb.log({f'train/top_{id+1}_acc':topkacc[id].item()/ct})

        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    if accelerator.is_local_main_process:
        for id, i in enumerate(top_3acc):
            wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
    if accelerator.is_local_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        print('Train Accuracy: {:.2f}%'.format(100 * correct / total))
        wandb.log({"train/epochacc": correct / total, "train/epochloss": epoch_loss})

    if (epoch + 1) % train_config["save_freq"] == 0:
        top_3acc = [0 for _ in range(3)]
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.eval()

        k_acc = [[] for i in range(5)]
        for batch_idx, data in enumerate(tqdm(test_loader)):

            if args.debug and batch_idx > 10:
                break

            with torch.no_grad():
                if batch_idx < 1:
                    acces = getkacc(model, data, head, max_length=5)
                    for i in range(len(acces)):
                        k_acc[i].append(acces[i])

                hidden_states = model.module.fusion_layer(data["hidden_states"])
                q_hidden_states = None
                unwarped_model.reset_step()
                for forward_idx in range(args.forward_num_total):
                    assert forward_idx == unwarped_model.current_step

                    predict, sample_hidden = model(hidden_states, input_ids=data["input_ids"],
                                                   attention_mask=data["attention_mask"],
                                                   q_hidden_states=q_hidden_states)
                    if q_hidden_states is None:
                        q_hidden_states = torch.cat([hidden_states[:, :1, :], predict[:, :-1, :]], dim=1)[None,
                        :, :, :]
                    else:
                        new_q_hidden_states = torch.cat([q_hidden_states[-1][:, :1, :], predict[:, :-1, :]], dim=1)[
                            None, :, :, :]
                        q_hidden_states = torch.cat([q_hidden_states, new_q_hidden_states], dim=0)

                target_head = head(data["target"])
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
                loss_mask = data["loss_mask"][:, :, None]

                if not args.use_adapter:
                    vloss, ploss, topk_loss, out_head = compute_loss(data["target"], target_p, predict, loss_mask)
                else:
                    vloss, ploss, topk_loss, out_head = compute_loss(data["target"], target_p, sample_hidden, loss_mask)

                loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss + train_config["topk_w"] * topk_loss

                _, predicted = torch.max(out_head, 2)
                _, target = torch.max(target_head, 2)
                ct = loss_mask.sum().item()
                cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
                out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
                target = target.view(-1)[loss_mask.view(-1) == 1]
                topkacc = top_accuracy(out_head, target, (1, 2, 3))
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]
                total += ct
                correct += cc
            epoch_loss += loss.item()
            num_batches += 1

        mean_acces = []
        for id, i in enumerate(k_acc):
            mean_acc = np.array(i).mean()
            mean_acc = torch.tensor(mean_acc).cuda()
            mean_acces.append(mean_acc)

        mean_acces = accelerator.gather_for_metrics(mean_acces)
        if accelerator.is_local_main_process:
            for id, i in enumerate(mean_acces):
                mean_acc = i.mean().item()
                wandb.log({f"test/{id}_acc": mean_acc})

        correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
        correct, total = accelerator.gather_for_metrics((correct, total))
        correct, total = correct.sum().item(), total.sum().item()
        top_3acc = accelerator.gather_for_metrics(top_3acc)
        if accelerator.is_local_main_process:
            for id, i in enumerate(top_3acc):
                wandb.log({f'test/top_{id + 1}_acc': i.sum().item() / total})
        epoch_loss /= num_batches
        if accelerator.is_local_main_process:
            print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
            print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
            wandb.log({"test/epochacc": correct / total, "test/epochloss": epoch_loss})
            accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}")
