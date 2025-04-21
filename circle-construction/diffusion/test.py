import torch
import argparse
import json

from load_model import load_model
from transformers import GPT2TokenizerFast
import sampling
from tqdm import tqdm

import os


def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_checkpoint_dir", default="", type=str)
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--add_vocab", type=str, default=None)
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    if args.add_vocab:
        with open(args.add_vocab, 'r') as file:
            added_tokens = json.load(file)
        tokenizer.add_tokens(added_tokens)
    else:
        added_tokens = []
    
    # Load the dataset
    with open(os.path.join(args.dataset, "test.json"), "r") as f:
        test_data = json.load(f)
    
    # List all files under the model checkpoint directory
    checkpoints = [os.path.join(args.model_checkpoint_dir, "checkpoints", f) for f in os.listdir(os.path.join(args.model_checkpoint_dir, "checkpoints"))]
    print(checkpoints)

    for checkpoint in checkpoints:
        device = torch.device('cuda')
        model, graph, noise = load_model(args.model_checkpoint_dir, checkpoint, added_tokens, device)
        # Create a checkpoint_dir for the current checkpoint
        checkpoint_dir = os.path.join(args.model_checkpoint_dir, "checkpoint_outputs", os.path.basename(checkpoint))
        if os.path.exists(checkpoint_dir):
            print(f"Skipping {checkpoint_dir} because it already exists")
            continue
        os.makedirs(checkpoint_dir, exist_ok=True)

        def generate_output(input_text):
            prefix_ids = tokenizer(input_text).input_ids
            # suffix_ids = tokenizer("<|endoftext|>").input_ids
            input_ids = prefix_ids
            input_locs = list(range(len(prefix_ids)))

            input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(1, 1)

            def proj_fun(x):
                x[:, input_locs] = input_ids
                return x
            
            sampling_fn = sampling.get_pc_sampler(
                graph, noise, (1, 29), 'analytic', args.steps, device=device, proj_fun=proj_fun
            )

            samples = proj_fun(sampling_fn(model))

            text_samples = tokenizer.batch_decode(samples)
            assert len(text_samples) == 1
            text_samples = text_samples[0].split("<|endoftext|>")[0]
            return text_samples
        
        all_items = []
        for sample in tqdm(test_data):
            item = {}
            item["input_text"] = sample["input_text"]
            item["target_text"] = sample["target_text"]
            item["type"] = sample["type"]

            output = generate_output(sample["input_text"])
            print(sample["input_text"])
            print(sample["target_text"])
            print(output)
            print()
            item["model_output"] = output
            all_items.append(item)
        
        # Save the results
        with open(os.path.join(checkpoint_dir, "all_items.json"), "w") as f:
            json.dump(all_items, f, indent=4)


if __name__=="__main__":
    main()