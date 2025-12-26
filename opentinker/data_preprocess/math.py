# Copyright 2024 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        default="./data",
        help="The local directory for the preprocessed dataset.",
    )
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="The local path to the raw dataset, if it exists.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="./data",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "DigitalLearningGmbH/MATH-lighteval"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = (
        "Let's think step by step and output the final answer within \\boxed{}."
    )

    # build map function
    def make_map_fn():
        def process_fn(example, idx):
            question = example.pop("problem") + " " + instruction_following

            answer = example.pop("solution")
            solution = extract_solution(answer)

            return {
                "prompt": [{"role": "user", "content": question}],
                "ground_truth": solution,
                "data_source": data_source,
            }

        return process_fn

    train_dataset = train_dataset.map(make_map_fn(), with_indices=True)
    test_dataset = test_dataset.map(make_map_fn(), with_indices=True)

    local_save_dir = args.local_dir or args.local_save_dir
    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    # --------------------
    # Write JSONL files
    # --------------------
    def write_jsonl(ds, path):
        with open(path, "w") as f:
            for item in ds:
                json.dump(item, f)
                f.write("\n")

    train_jsonl = os.path.join(local_dir, "train.jsonl")
    test_jsonl = os.path.join(local_dir, "test.jsonl")

    print(f"Writing JSONL to {train_jsonl} and {test_jsonl}")
    write_jsonl(train_dataset, train_jsonl)
    write_jsonl(test_dataset, test_jsonl)

    # Save first example for reference
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(train_dataset[0], f, indent=2)
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(test_dataset[0], f, indent=2)

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
