import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from utils.util import get_all_model_names


def plot_compass_positions(
    df,
    extreme_positions,
    n=5,
    bins="log",
    gridsize=30,
    label_gap=1.5,
    label_margin=0.15,
    tick_interval=2,
    label_size=10,
    tick_size=8,
    colorbar_size=8,
    origin_lines_alpha=0.8,
):
    # Extract coordinates
    x = [pos[0] for pos in df["compass_position"]]
    y = [pos[1] for pos in df["compass_position"]]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Custom formatter function for axis ticks
    def custom_formatter(x, p):
        if x == -10:
            return ""
        if abs(x) >= 1000:
            return f"{int(x/1000)}k"
        return str(int(x))

    # Create the hexbin plot with borders
    hb = ax.hexbin(
        x,
        y,
        gridsize=gridsize,
        cmap="YlOrRd",
        mincnt=1,
        bins=bins,
        extent=(-10, 10, -10, 10),
        edgecolors=(0, 0, 0, 0.3),
        linewidths=0.1,
    )

    # Set the axis limits and ticks
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # Set tick intervals
    ticks = np.arange(-10, 11, tick_interval)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # Set minor grid lines
    grid_lines = np.arange(-10, 11, 1)
    ax.set_xticks(grid_lines, minor=True)
    ax.set_yticks(grid_lines, minor=True)

    # Show all spines
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.2)

    # Apply custom formatter
    ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))

    # Set tick label size
    ax.tick_params(axis="both", labelsize=tick_size)

    # Remove default axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Adjust margins
    margin_size = label_gap / 20
    plt.subplots_adjust(left=0.15 + margin_size, bottom=0.15 + margin_size)

    # Add labels
    label_pos = 10 + label_gap
    ax.text(
        -10,
        -label_pos,
        "← Left",
        ha="left",
        va="top",
        transform=ax.transData,
        fontsize=label_size,
    )
    ax.text(
        10,
        -label_pos,
        "Right →",
        ha="right",
        va="top",
        transform=ax.transData,
        fontsize=label_size,
    )
    ax.text(
        -label_pos,
        10,
        "Authorit. →",
        ha="right",
        va="top",
        rotation=90,
        transform=ax.transData,
        fontsize=label_size,
    )
    ax.text(
        -label_pos,
        -10,
        "← Libert.",
        ha="right",
        va="bottom",
        rotation=90,
        transform=ax.transData,
        fontsize=label_size,
    )

    # Add the -10 label
    ax.text(
        -10.3,
        -10.3,
        "-10",
        ha="right",
        va="top",
        transform=ax.transData,
        fontsize=tick_size,
    )

    # Add grid lines
    ax.grid(True, which="major", linestyle="-", alpha=0.7)
    ax.grid(True, which="minor", linestyle="--", alpha=0.7)

    # Add diagonal lines
    ax.plot([-10, 10], [-10, 10], "--", color="gray", alpha=0.3)
    ax.plot([-10, 10], [10, -10], "--", color="gray", alpha=0.3)

    # Add center lines
    ax.axhline(y=0, color="k", linestyle="-", alpha=origin_lines_alpha, linewidth=1.5)
    ax.axvline(x=0, color="k", linestyle="-", alpha=origin_lines_alpha, linewidth=1.5)

    # Add colorbar
    cbar_ax = fig.add_axes([0.95, 0.15, 0.03, 0.75])
    cbar = fig.colorbar(hb, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=colorbar_size)

    # Plot extreme points
    colors = ["red", "blue", "green", "purple"]
    for (quadrant, positions), color in zip(extreme_positions.items(), colors):
        if positions:
            x_pos = [pos[2][0] for pos in positions]
            y_pos = [pos[2][1] for pos in positions]
            ax.scatter(
                x_pos,
                y_pos,
                color=color,
                s=100,
                label=f"{quadrant}",
                zorder=5,
                alpha=0.5,
            )

    return fig


def plot_model(model: str, n_corner_personas:int, n_left_right_personas:int, use_center:bool = False):
    MODEL = model.split("/")[-1]
    compass_file_name = "final_compass"
    suffix = "_centered" if use_center else ""

    extreme_compass_dir = f"images/extreme_compass/{MODEL}"
    if not os.path.exists(extreme_compass_dir):
        os.makedirs(extreme_compass_dir)

    print()
    print("=" * 70)
    print(f"Model: {model}")
    df_path = f"../extension-llm-political-personas/results/{MODEL}/base/{compass_file_name}.pqt"
    print(f"Reading data from: {df_path}")
    df = pd.read_parquet(df_path)
    print(f"DataFrame shape: {df.shape}")
    assert df.shape[0] == 12400000
    print("=" * 70)
    print()

    print()
    print("=" * 70)
    print("Loading extreme positions data...")
    with open(
        f"data/results/extreme_pos_personas/{MODEL}/extreme_pos_corners_{n_corner_personas}{suffix}.pkl", "rb"
    ) as f:
        extreme_pos = pickle.load(f)
    with open(
        f"data/results/extreme_pos_personas/{MODEL}/extreme_pos_left_right_{n_left_right_personas}{suffix}.pkl", "rb"
    ) as f:
        extreme_pos_l_r = pickle.load(f)
    print("=" * 70)
    print()

    print()
    print("=" * 70)
    print("Generating and saving 'corners' plot...")
    _ = plot_compass_positions(
        df,
        extreme_pos,
        bins="log",
        gridsize=40,
        label_gap=2.5,
        label_margin=0.6,
        tick_interval=5,
        label_size=31,
        tick_size=28,
        colorbar_size=28,
        origin_lines_alpha=0.4,
    )

    plt.savefig(
        os.path.join(extreme_compass_dir, f"{MODEL}_extreme_corners_{n_corner_personas}{suffix}.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
    print("=" * 70)
    print()

    print()
    print("=" * 70)
    print("Generating and saving 'left-right' plot...")
    _ = plot_compass_positions(
        df,
        extreme_pos_l_r,
        bins="log",
        gridsize=40,
        label_gap=2.5,
        label_margin=0.6,
        tick_interval=5,
        label_size=31,
        tick_size=28,
        colorbar_size=28,
        origin_lines_alpha=0.4,
    )

    plt.savefig(
        os.path.join(extreme_compass_dir, f"{MODEL}_extreme_left_right_{n_left_right_personas}{suffix}.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
    print("=" * 70)
    print()


# ======================================= MAIN =======================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=int, default=-1)
    parser.add_argument(
        "--n_corner_personas",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--n_left_right_personas",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--use_center",
        action="store_true",
        help="Needed to read and store the correct file",
    )
    args = parser.parse_args()
    models = get_all_model_names()

    if args.model == -1:
        # loop on all models and plot
        for model_index, model in enumerate(models):
            print(f"Processing model {model_index + 1}/{len(models)}: {model}")
            plot_model(model, 
                       n_corner_personas=args.n_corner_personas,
                       n_left_right_personas=args.n_left_right_personas, use_center=args.use_center)
    else:
        model = models[args.model]
        plot_model(model,
                   n_corner_personas=args.n_corner_personas,
                   n_left_right_personas=args.n_left_right_personas, use_center=args.use_center)


if __name__ == "__main__":
    main()
