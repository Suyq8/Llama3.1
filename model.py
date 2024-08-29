import os
from argparse import Namespace
from typing import List, Optional
import time

import torch
from tqdm import tqdm

from module import KVCache, Transformer
from tokenizer import Tokenizer
from quantize import WeightOnlyInt8QuantHandler


class Llama:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def build(args: Namespace) -> "Llama":
        assert (
            1 <= args.max_seq_len <= 8192
        ), f"max_seq_len must be between 1 and 8192, got {args.max_seq_len}."
        assert os.path.exists(
            args.ckpt_path
        ), f"Checkpoint file '{args.ckpt_path}' does not exist."
        assert os.path.isfile(
            args.tokenizer_path
        ), f"Tokenizer file '{args.tokenizer_path}' does not exist."

        t0 = time.time()
        checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)

        tokenizer = Tokenizer(args.tokenizer_path)
        assert args.vocab_size == tokenizer.n_words

        with torch.device("meta"):
            model = Transformer(args)

        if "int8" in str(args.ckpt_path):
            print("Using int8 weight-only quantization!")
            simple_quantizer = WeightOnlyInt8QuantHandler(model)
            model = simple_quantizer.convert_for_runtime()

        model.load_state_dict(checkpoint, strict=False, assign=True)
        model.to(args.device, dtype=torch.bfloat16)

        torch.compile(
            model.forward_inference,
        )
        t1 = time.time()
        print(f"Model loaded in {t1 - t0:.2f} seconds")

        return Llama(model, tokenizer)

    def sample_top_p(self, probs: torch.Tensor, top_p: float) -> torch.Tensor:
        # probs: (batch_size, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    #@torch.compile(dynamic=True, fullgraph=True)
    def generate_one_next_token(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        logits = self.model.forward_inference(x, start_pos)  # (batch_size, seq_len, vocab_size)

        args = self.model.args
        if args.temperature > 0:
            probs = torch.softmax(logits[:, -1] / args.temperature, dim=-1)
            next_token = self.sample_top_p(probs, args.top_p).flatten()
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1).flatten()

        return next_token

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[List[int]],
        max_gen_len: Optional[int] = None,
    ) -> List[List[int]]:
        args = self.model.args
        batch_size = len(prompts)
        assert (
            batch_size <= args.max_batch_size
        ), f"batch size must be <= {args.max_batch_size}, got {batch_size}."

        max_prompt_len = max(len(p) for p in prompts)
        min_prompt_len = min(len(p) for p in prompts)
        assert (
            max_prompt_len <= args.max_seq_len
        ), f"max prompt length must be <= {args.max_seq_len}, got {max_prompt_len}."

        total_len = min(args.max_seq_len, max_gen_len + max_prompt_len)

        # set kv_cache for each layer
        for layer in self.model.layers:
            layer.attention.kv_cache = KVCache(
                batch_size=args.max_batch_size,
                max_seq_len=args.max_seq_len,
                n_kv_head=args.n_kv_head,
                head_dim=args.dim // args.n_head,
                dtype=torch.bfloat16,
                device=layer.attention.wo.weight.device,
            )
        self.model.eval()

        pad_id = self.tokenizer.pad_id
        tokens = torch.full(
            (batch_size, total_len), pad_id, dtype=torch.long, device=args.device
        )
        for i, p in enumerate(prompts):
            tokens[i, : len(p)] = torch.tensor(p, dtype=torch.long, device=args.device)
        #print(tokens.device, self.model.tok_embeddings.weight.device)

        prev_pos = 0
        eos_reached = torch.tensor([False] * batch_size, device=args.device)
        input_text_mask = tokens == pad_id
        stop_tokens = torch.tensor(self.tokenizer.stop_tokens, device=args.device)
        for cur_pos in tqdm(range(min_prompt_len, total_len)):
            next_token = self.generate_one_next_token(
                tokens[:, prev_pos:cur_pos], prev_pos
            )

            next_token = torch.where(
                input_text_mask[:, cur_pos], next_token, tokens[:, cur_pos]
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (
                torch.isin(next_token, stop_tokens) & input_text_mask[:, cur_pos]
            )

            prev_pos = cur_pos

            if torch.all(eos_reached):
                break

        output_tokens = []
        for i in range(batch_size):
            start_pos = len(prompts[i])
            idx = torch.where(torch.isin(tokens[i, start_pos:], stop_tokens))[0]
            if len(idx) > 0:
                end_pos = idx[0]
            else:
                end_pos = total_len
            output_tokens.append(tokens[i, start_pos:end_pos].tolist())

        # clear kv_cache
        # for layer in self.model.layers:
        #     layer.attention.kv_cache = None

        return output_tokens

    def text_completion(
        self,
        prompts: List[str],
        max_gen_len: Optional[int] = None,
    ) -> List[str]:
        if max_gen_len is None:
            max_gen_len = self.model.args.max_seq_len - 1

        tokens = list(self.tokenizer.encode(p, bos=True, eos=False) for p in prompts)

        generated_tokens = self.generate(tokens, max_gen_len)

        return list(
            self.tokenizer.decode(t) for t in generated_tokens
        )
