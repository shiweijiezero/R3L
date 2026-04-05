import glob
import json
import os
import random

random.seed(42)


# FIX 1: 将默认值改为 None
def create_dataset_files(output_dir, train_size=None, test_size=None):
    alfworld_data_root = "/path/to/alfworld/json_2.1.1"

    train_game_files = glob.glob(os.path.join(alfworld_data_root, "train/*/*/game.tw-pddl"))
    test_game_files = glob.glob(os.path.join(alfworld_data_root, "valid_seen/*/*/game.tw-pddl"))

    train_game_files = sorted([os.path.abspath(file) for file in train_game_files])
    test_game_files = sorted([os.path.abspath(file) for file in test_game_files])

    print(f"Total train game files found: {len(train_game_files)}")
    print(f"Total test game files found: {len(test_game_files)}")

    # FIX 2: 如果参数为 None，则使用全部文件
    if train_size is None:
        train_size = len(train_game_files)
        print(f"train_size not set, defaulting to all: {train_size}")
    if test_size is None:
        test_size = len(test_game_files)
        print(f"test_size not set, defaulting to all: {test_size}")

    # check sizes
    assert train_size <= len(
        train_game_files
    ), f"train_size {train_size} > available {len(train_game_files)}"
    assert test_size <= len(
        test_game_files
    ), f"test_size {test_size} > available {len(test_game_files)}"

    # 随机采样
    selected_train_files = random.sample(train_game_files, train_size)
    selected_test_files = random.sample(test_game_files, test_size)

    os.makedirs(output_dir, exist_ok=True)

    def _create_data_list(game_files):
        data = []
        for game_file_path in game_files:
            data.append({"game_file": game_file_path, "target": ""})
        return data

    train_data = _create_data_list(selected_train_files)
    test_data = _create_data_list(selected_test_files)

    dataset_dict = {"train": train_data, "test": test_data}

    for split, data in dataset_dict.items():
        output_file = os.path.join(output_dir, f"{split}.jsonl")
        with open(output_file, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    dataset_info = {
        "citation": "",
        "description": "Custom dataset",
        "splits": {
            "train": {"name": "train", "num_examples": len(train_data)},
            "test": {"name": "test", "num_examples": len(test_data)},
        },
    }

    with open(os.path.join(output_dir, "dataset_dict.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Created dataset with {len(train_data)} train and {len(test_data)} test examples.")


if __name__ == "__main__":
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = f"{current_file_dir}/alfworld_data"

    # 1. 使用全部数据 (train=3553, test=140)
    print("--- Creating full dataset ---")
    create_dataset_files(output_dir)

    # # 2. 仍然可以指定特定大小
    # print("\n--- Creating subset dataset ---")
    # create_dataset_files(output_dir, train_size=2953, test_size=100)
    #
    # # 3. 只指定训练集大小 (测试集将使用全部)
    # print("\n--- Creating partial subset dataset ---")
    # create_dataset_files(output_dir, train_size=100)
