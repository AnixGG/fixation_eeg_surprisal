"""Plot the fraction of MATLAB fixation slots matched in each NR block."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=here / "full_blocks.csv")
    parser.add_argument("--out", type=Path, default=here / "coverage_generated.png")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    blocks = pd.read_csv(args.table)
    counts = blocks[["MATCHED", "AMBIGUOUS", "UNMATCHED", "INVALID"]].sum(axis=1)
    rates = blocks.assign(percent=100 * blocks.MATCHED / counts).pivot(
        index="participant", columns="run", values="percent")
    fig, ax = plt.subplots(figsize=(8, 6.6), layout="constrained")
    image = ax.imshow(rates, vmin=0, vmax=100, cmap="viridis", aspect="auto")
    ax.set(xticks=range(len(rates.columns)), xticklabels=rates.columns,
           yticks=range(len(rates)), yticklabels=rates.index, xlabel="NR run")
    for row in blocks.itertuples():
        if not row.clock_accepted:
            ax.text(rates.columns.get_loc(row.run), rates.index.get_loc(row.participant),
                    "×", ha="center", va="center", color="white", fontsize=12)
    ax.set_title("Word-linked fixations matched to the physical event stream", loc="left", pad=14)
    ax.text(0, -0.105, "×  Run clock rejected. Grey cells have no saved fixation slots.",
            transform=ax.transAxes, fontsize=9)
    ax.set_facecolor("#a3a6a8")
    fig.colorbar(image, ax=ax, label="Matched / saved MATLAB fixation slots (%)")
    fig.savefig(args.out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
