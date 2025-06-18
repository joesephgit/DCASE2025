import os
import argparse
import importlib
import importlib.resources as pkg_resources
import pandas as pd
import torch
import torch.nn.functional as F
import json
from torch.hub import download_url_to_file
from sklearn.metrics import accuracy_score


# Dataset config
dataset_config = {
    "split_url": "https://github.com/CPJKU/dcase2024_task1_baseline/releases/download/files/",
    "test_split_csv": "test.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a DCASE Task 1 submission.")
    parser.add_argument("--submission_name", type=str, required=True)
    parser.add_argument("--submission_index", type=int, required=True)
    parser.add_argument("--dev_set_dir", type=str, required=True)
    parser.add_argument("--eval_set_dir", type=str, required=True)
    return parser.parse_args()


def download_split_file(resource_dir: str, split_name: str) -> str:
    os.makedirs(resource_dir, exist_ok=True)
    split_path = os.path.join(resource_dir, split_name)
    if not os.path.isfile(split_path):
        print(f"Downloading {split_name} to {split_path} ...")
        download_url_to_file(dataset_config["split_url"] + split_name, split_path)
    return split_path


def load_test_split(dataset_dir: str, resource_pkg: str) -> pd.DataFrame:
    meta_csv = os.path.join(dataset_dir, "meta.csv")

    try:
        with pkg_resources.path(resource_pkg, "test.csv") as test_csv_path:
            test_csv_file = str(test_csv_path)
    except FileNotFoundError:
        print("test.csv not found in package resources. Downloading ...")
        resource_dir = os.path.join(os.path.dirname(__file__), resource_pkg.replace('.', '/'), "resources")
        test_csv_file = download_split_file(resource_dir, dataset_config["test_split_csv"])

    df_meta = pd.read_csv(meta_csv, sep="\t")
    df_test = pd.read_csv(test_csv_file, sep="\t").drop(columns=["scene_label"], errors="ignore")
    df_test = df_test.merge(df_meta, on="filename")

    return df_test


def run_evaluation(args):
    # --- Load module ---
    module_path = f"{args.submission_name}.{args.submission_name}_{args.submission_index}"
    print(f"Importing inference module: {module_path}")
    api = importlib.import_module(module_path)

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")

    # --- Load test data ---
    print("Loading test split ...")
    df_test = load_test_split(args.dev_set_dir, f"{args.submission_name}.resources")
    file_paths = [os.path.join(args.dev_set_dir, fname) for fname in df_test["filename"]]
    device_ids = df_test["source_label"].tolist()
    scene_labels = df_test["scene_label"].tolist()

    print("Running test set predictions ...")
    predictions, class_order = api.predict(
        file_paths=file_paths,
        device_ids=device_ids,
        model_file_path=None,
        use_cuda=use_cuda
    )

    # Map ground truth scene labels to class indices based on class_order
    label_to_idx = {label: idx for idx, label in enumerate(class_order)}
    true_labels = [label_to_idx[label] for label in scene_labels]

    # Compute accuracy
    pred_labels = [pred.argmax().item() for pred in predictions]
    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

    # --- Load evaluation data ---
    print("\nLoading evaluation set ...")
    df_eval = pd.read_csv(os.path.join(args.eval_set_dir, "evaluation_setup", "fold1_test.csv"), sep="\t")
    eval_file_paths = [os.path.join(args.eval_set_dir, fname) for fname in df_eval["filename"]]
    eval_device_ids = df_eval["device_id"].tolist()

    print("Running evaluation set predictions ...")
    eval_predictions, eval_class_order = api.predict(
        file_paths=eval_file_paths,
        device_ids=eval_device_ids,
        model_file_path=None,
        use_cuda=use_cuda
    )

    assert eval_class_order == class_order, "Class order mismatch between test and evaluation prediction"

    # --- Format and save submission ---
    output_dir = os.path.join("predictions", f"{args.submission_name}_{args.submission_index}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving predictions to: {output_dir}/output.csv")
    all_probs = torch.stack(eval_predictions)
    all_probs = F.softmax(all_probs, dim=1)
    predicted_labels = [class_order[i] for i in torch.argmax(all_probs, dim=1)]

    submission = pd.DataFrame({
        "filename": df_eval["filename"],
        "scene_label": predicted_labels
    })
    for i, label in enumerate(class_order):
        submission[label] = all_probs[:, i].tolist()

    submission.to_csv(os.path.join(output_dir, "output.csv"), sep="\t", index=False)

    # --- Save model weights and info ---
    model = api.load_model()
    torch.save(model.model.state_dict(), os.path.join(output_dir, "model_state_dict.pt"))

    info = {
        "Test Accuracy": round(acc * 100, 2)
    }
    with open(os.path.join(output_dir, "test_accuracy.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n✅ Submission ready: {output_dir}/output.csv")


def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
