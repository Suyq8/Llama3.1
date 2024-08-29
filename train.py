import argparse
import glob
from math import ceil
import time
from typing import List, Tuple

import numpy as np
import torch

from model import Llama
from module import Transformer


def _peek_data_shard(filename: str) -> int:
    data = np.memmap(filename, dtype=np.uint32, mode="r")
    ntok = len(data)

    return ntok  # for now just return the number of tokens


def _load_data_shard(filename: str) -> np.ndarray:
    tokens = np.fromfile(filename, dtype=np.uint32)
    return tokens


class DistributedShardedDataLoader:
    """
    This DataLoader is both:
    - distributed (works correctly in case of multiple processes in DDP)
    - sharded (supports datasets that are broken up into multiple data shards)
    It is not *permuted*, meaning that it itearates over the data in the order
    of the dataset on disk, so the user should make sure to shuffle their examples
    during the creation of their data shards for best performance.
    """

    def __init__(
        self,
        filename_pattern: str,
        B: int,
        T: int,
        process_rank: int,
        num_processes: int,
    ):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B  # batch size
        self.T = T  # sequence length

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert (
            len(self.files) > 0
        ), f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            print(f"DataLoader: {fname} has {shard_ntok:,} tokens")
            print(num_processes * B * T + 1)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print(
            f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files"
        )

        # kick things off
        self.current_shard = None
        self.length = ntok_total // (B * T * num_processes)
        self.reset()

    def __len__(self):
        return self.length

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf, dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds advance the shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y


class CosineSchedulerWithWarmup:
    def __init__(
        self, optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = optimizer.param_groups[0]["lr"]
        self.min_lr = self.max_lr * 0.1

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps

        if step >= self.max_steps:
            return self.min_lr

        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + np.cos(np.pi * progress)
        )  # half cycle cosine

    def step(self, step: int):
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


def get_optimizer(
    model: Transformer, weight_decay: float, learning_rate: float, device_type: str
) -> torch.optim.Optimizer:
    # start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    # Create AdamW optimizer and use the fused version if it is available
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=device_type == "cuda",
    )

    return optimizer


def main(main_args: argparse.Namespace):
    # load the val data shard
    train_loader = DistributedShardedDataLoader(
        filename_pattern=f"{main_args.data_dir}/train.bin",
        B=main_args.max_batch_size,
        T=main_args.max_seq_len,
        process_rank=0,
        num_processes=1,
    )
    val_loader = DistributedShardedDataLoader(
        filename_pattern=f"{main_args.data_dir}/val.bin",
        B=main_args.max_batch_size//4,
        T=main_args.max_seq_len,
        process_rank=0,
        num_processes=1,
    )

    llama = Llama.build(main_args)

    total_batch_size = 2**16  # ~65k tokens
    grad_accum_steps = total_batch_size // (
        main_args.max_batch_size * main_args.max_seq_len
    )
    max_steps = ceil(len(train_loader) / grad_accum_steps) * main_args.epoch

    # super simple training loop to start
    model = llama.model
    optimizer = get_optimizer(
        model, learning_rate=6e-4, weight_decay=0.1, device_type=main_args.device
    )
    scheduler = CosineSchedulerWithWarmup(
        optimizer, warmup_steps=100, max_steps=max_steps
    )

    for epoch in range(main_args.epoch):
        t0 = time.time()
        total_loss = 0
        model.train()
        for step in range(max_steps):
            optimizer.zero_grad()
            total_loss = 0
            model.train()
            for _ in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                x, y = x.to(main_args.device), y.to(main_args.device)
                with torch.autocast(device_type=main_args.device, dtype=torch.bfloat16):
                    loss = model.forward_loss(x, y) / grad_accum_steps
                loss.backward()
                total_loss += loss.item()
            optimizer.step()
            scheduler.step(epoch * grad_accum_steps + step)

            if step % 10 == 0:
                print(f"epoch {epoch}, step {step}, loss: {total_loss.item():.6f}")
                val_loss = 0
                model.eval()
                for _ in range(len(val_loader)):
                    x, y = val_loader.next_batch()
                    x, y = x.to(main_args.device), y.to(main_args.device)
                    with torch.no_grad():
                        val_loss += model.forward_loss(x, y).item()
                val_loss /= len(val_loader)
                print(f"val loss: {val_loss.item():.6f}")

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print(
            f"epoch {epoch}, loss: {total_loss.item():.6f}, time: {time.time() - t0:.4f}s, norm: {norm:.4f}"
        )

    # and now generate
    model.eval()
    prompts: List[str] = [
        "Once upon a time",
        "One day",
        "My queen and son are gone to France for aid",
        "Thus have I politicly begun my reign,",
    ]

    t0 = time.time()
    results = llama.text_completion(prompts, main_args.max_gen_len)

    t1 = time.time()
    print(f"Generated in {t1 - t0:.2f} seconds")
    for prompt, result in zip(prompts, results):
        print(prompt, end="")  # AK: change end="\n" to end=""
        print(f"{result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the model checkpoint to be quantized.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the tokenizer model",
    )

    parser.add_argument("--vocab_size", type=int, default=128256)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_batch_size", type=int, default=32)
    parser.add_argument("--n_layer", type=int, default=32)
    parser.add_argument("--n_head", type=int, default=32)
    parser.add_argument("--n_kv_head", type=int, default=8)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--ffn_dim_multiplier", type=float, default=1.3)
    parser.add_argument("--norm_eps", type=float, default=1e-5)
    parser.add_argument("--rope_theta", type=float, default=500000)
    parser.add_argument("--use_scaled_rope", type=bool, default=True)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--multiple_of", type=int, default=1024)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--use_flash", type=bool, default=False)

    args = parser.parse_args()
    main(args)
