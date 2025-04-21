#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
import torch_dwn as dwn
import openml
from sklearn.model_selection import train_test_split


def load_and_prepare(dataset_id: int, thermometer_bits: int = 200, test_size: float = 0.2, random_state: int = 42):
    """Fetches an OpenML dataset, splits it, and applies thermometer encoding."""
    ds = openml.datasets.get_dataset(dataset_id)
    X_df, y_df, _, _ = ds.get_data(dataset_format='dataframe',
                                   target=ds.default_target_attribute)

    X = X_df.values.astype(np.float32)
    labels_unique = y_df.unique().tolist()
    y = y_df.map(lambda v: labels_unique.index(v)).to_numpy(dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    therm = dwn.DistributiveThermometer(thermometer_bits).fit(X_train)
    X_train = therm.binarize(X_train).reshape(X_train.shape[0], -1)
    X_test  = therm.binarize(X_test) .reshape(X_test.shape[0],  -1)

    # Convert labels to torch tensors
    y_train = torch.from_numpy(y_train)
    y_test  = torch.from_numpy(y_test)

    return (X_train, y_train), (X_test, y_test), labels_unique


def export_jsc_format(
    data: np.ndarray,
    labels: torch.Tensor,
    mapping: np.ndarray,
    output_dir: Path,
    max_samples: int = None
):
    """
    Exports `data` and `labels` into 'dataset.txt' and 'predictions.txt'
    using a feature-index mapping.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    n_samples = data.shape[0] if max_samples is None else min(max_samples, data.shape[0])
    # Select and flatten features according to mapping
    # mapping: array of column indices, shape (n_features_out,)
    # data: shape (n_samples, n_features_in)
    selected = data[:n_samples, mapping].astype(np.int64)

    # Write dataset.txt
    np.savetxt(
        output_dir / "dataset.txt",
        selected,
        fmt="%d",
        delimiter="",
    )

    # Write predictions.txt
    np.savetxt(
        output_dir / "predictions.txt",
        labels[:n_samples].numpy().reshape(-1, 1),
        fmt="%d",
    )

    print(f"Exported {n_samples} samples to '{output_dir}'.")


def main():
    p = argparse.ArgumentParser(description="Export OpenML dataset to JSC format")
    p.add_argument("-i", "--input",       required=True, help="CSV mapping file (semicolon-separated indices)")
    p.add_argument("-n", "--num_samples", type=int,   default=None, help="Max number of samples to export")
    p.add_argument("-d", "--dataset",      choices=("train", "test"), required=True,
                   help="Which split to export")
    p.add_argument("-o", "--output_dir",   required=True, help="Directory to write output files")
    p.add_argument("--dataset_id",         type=int, default=42468,
                   help="OpenML dataset ID (default: 42468)")
    p.add_argument("--bits",               type=int, default=200,
                   help="Number of thermometer bits (default: 200)")
    args = p.parse_args()

    # Load and preprocess
    (X_train, y_train), (X_test, y_test), _ = load_and_prepare(
        dataset_id=args.dataset_id,
        thermometer_bits=args.bits
    )
    X, y = (X_train, y_train) if args.dataset == "train" else (X_test, y_test)

    # Load mapping
    # expects a single row semicolon-delimited CSV of integer indices
    mapping = pd.read_csv(args.input, sep=";", header=None).iloc[0].to_numpy(dtype=int)

    export_jsc_format(
        data=X.cpu().numpy(),
        labels=y,
        mapping=mapping,
        output_dir=Path(args.output_dir),
        max_samples=args.num_samples
    )


if __name__ == "__main__":
    main()