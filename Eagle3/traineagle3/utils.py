__all__ = ["scan_data"]

from collections import Counter

import torch
from datasets import Dataset
from loguru import logger
from tqdm import tqdm


def scan_data(
        dataset: Dataset,
        draft_vocab_size: int = 32000,
        vocab_size: int = 128256
) -> None:
    logger.info(f"dataset info:\n{dataset}")

    token_dict = Counter()
    input_ids = dataset["input_ids"]
    loss_mask = dataset["loss_mask"]
    for i in tqdm(range(len(input_ids))):
        ids = input_ids[i][0].tolist()
        mask = loss_mask[i][0].tolist()
        for j in range(len(ids)):
            if mask[j] == 1:
                token_dict[ids[j]] += 1

    total_frequency = sum(token_dict.values())
    top_N = token_dict.most_common(draft_vocab_size)
    top_N_frequency_sum = sum(freq for key, freq in top_N)
    top_N_ratio = top_N_frequency_sum / total_frequency
    logger.info(f"top {draft_vocab_size} token frequency ratio: {top_N_ratio:.2%}")

    used_tokens = [key for key, freq in top_N]
    used_tokens.sort()
    d2t = [used_tokens[i] - i for i in range(len(used_tokens))]
    t2d = [i in used_tokens for i in range(vocab_size)]
    d2t = torch.tensor(d2t)
    t2d = torch.tensor(t2d)
    cache = {
        "d2t": d2t,
        "t2d": t2d
    }
    torch.save(cache, "cache.pt")
    logger.info("cache is saved")
