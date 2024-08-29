import argparse
from typing import List
from model import Llama

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the tokenizer model",
    )
    parser.add_argument("--vocab_size", type=int, default=128256)
    parser.add_argument("--device", type=str, default="cuda")
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
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--multiple_of", type=int, default=1024)
    parser.add_argument("--use_flash", type=bool, default=False)

    args = parser.parse_args()

    llama = Llama.build(args)

    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,

        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]

    results = llama.text_completion(prompts)

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result}")
        print("\n==================================\n")
