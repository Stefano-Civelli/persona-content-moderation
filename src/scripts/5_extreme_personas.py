import os
import pandas as pd
import pandas as pd
import numpy as np
import pickle
import argparse
from utils.util import get_all_model_names


# extreme pos 4 corner
def find_extreme_positions(df, n=5, weight_diagonal=0.5, use_center=False):
    # Extract coordinates
    df = df.copy()
    df["x"] = [pos[0] for pos in df["compass_position"]]
    df["y"] = [pos[1] for pos in df["compass_position"]]

    # Calculate center of distribution if requested
    if use_center:
        center_x = df["x"].mean()
        center_y = df["y"].mean()
        print(f"Using distribution center as origin: ({center_x:.4f}, {center_y:.4f})")
        
        # Adjust coordinates relative to center
        df["x_centered"] = df["x"] - center_x
        df["y_centered"] = df["y"] - center_y
        
        # Calculate distance from center
        df["distance_origin"] = np.sqrt(df["x_centered"] ** 2 + df["y_centered"] ** 2)
        
        # Use centered coordinates for quadrant determination
        x_col, y_col = "x_centered", "y_centered"
    else:
        # Calculate distance from origin (0,0)
        df["distance_origin"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
        
        # Use original coordinates
        x_col, y_col = "x", "y"

    def distance_to_diagonal(x, y, negative=False):
        if negative:
            return abs(y + x) / np.sqrt(2)
        else:
            return abs(y - x) / np.sqrt(2)

    # Initialize dictionary for extreme positions
    extreme_positions = {
        "top_right": [],
        "top_left": [],
        "bottom_right": [],
        "bottom_left": [],
    }

    # Process each quadrant
    quadrants = {
        "top_right": {"x_cond": f"{x_col} > 0", "y_cond": f"{y_col} > 0", "negative_diag": False},
        "top_left": {"x_cond": f"{x_col} < 0", "y_cond": f"{y_col} > 0", "negative_diag": True},
        "bottom_right": {"x_cond": f"{x_col} > 0", "y_cond": f"{y_col} < 0", "negative_diag": True},
        "bottom_left": {"x_cond": f"{x_col} < 0", "y_cond": f"{y_col} < 0", "negative_diag": False},
    }

    for quadrant, conditions in quadrants.items():
        quadrant_df = df.query(
            f"{conditions['x_cond']} & {conditions['y_cond']}"
        ).copy()

        if not quadrant_df.empty:
            quadrant_df["distance_diagonal"] = quadrant_df.apply(
                lambda row: distance_to_diagonal(
                    row[x_col], row[y_col], conditions["negative_diag"]
                ),
                axis=1,
            )

            quadrant_df["norm_dist_origin"] = (
                quadrant_df["distance_origin"] / quadrant_df["distance_origin"].max()
            )
            quadrant_df["norm_dist_diagonal"] = (
                quadrant_df["distance_diagonal"]
                / quadrant_df["distance_diagonal"].max()
            )

            quadrant_df["extreme_score"] = (1 - weight_diagonal) * quadrant_df[
                "norm_dist_origin"
            ] + weight_diagonal * (1 - quadrant_df["norm_dist_diagonal"])

            extreme = (
                quadrant_df.sort_values("extreme_score", ascending=False)
                .drop_duplicates("persona_id")
                .head(n)
            )

            extreme_positions[quadrant] = list(
                zip(
                    extreme["persona_id"],
                    extreme["cleaned_persona"],
                    extreme["compass_position"],
                )
            )

    return extreme_positions


def find_left_right_extreme_positions(df, n=5, use_center=False):
    # Extract coordinates
    df = df.copy()
    df["x"] = [pos[0] for pos in df["compass_position"]]
    df["y"] = [pos[1] for pos in df["compass_position"]]

    # Calculate center of distribution if requested
    if use_center:
        center_x = df["x"].mean()
        center_y = df["y"].mean()
        print(f"Using distribution center for left/right: ({center_x:.4f}, {center_y:.4f})")
        
        # Adjust coordinates relative to center
        df["x_centered"] = df["x"] - center_x
        x_col = "x_centered"
    else:
        x_col = "x"

    # Initialize dictionary for extreme positions
    extreme_positions = {"leftmost": [], "rightmost": []}

    # Get leftmost positions
    leftmost = df.sort_values(x_col, ascending=True).drop_duplicates("persona_id").head(n)
    extreme_positions["leftmost"] = list(
        zip(leftmost["persona_id"], leftmost["persona"], leftmost["compass_position"])
    )

    # Get rightmost positions
    rightmost = (
        df.sort_values(x_col, ascending=False).drop_duplicates("persona_id").head(n)
    )
    extreme_positions["rightmost"] = list(
        zip(
            rightmost["persona_id"], rightmost["persona"], rightmost["compass_position"]
        )
    )

    return extreme_positions


def process(model: str, n_corner_personas: int, n_left_right_personas: int, use_center: bool):
    MODEL = model.split("/")[-1]
    compass_file_name = "final_compass"

    df_path = f"../extension-llm-political-personas/results/{MODEL}/base/{compass_file_name}.pqt"

    print()
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Reading data from: {df_path}")
    df = pd.read_parquet(df_path)
    print(f"DataFrame shape: {df.shape}")
    if use_center:
        print("Using distribution center as origin for diagonal calculations")
    else:
        print("Using (0,0) as origin for diagonal calculations")
    print("=" * 70)
    print()

    assert df.shape[0] == 12400000

    print()
    print("=" * 70)
    print("Finding extreme corner positions...")
    extreme_pos = find_extreme_positions(df, n=n_corner_personas, weight_diagonal=0.4, use_center=use_center)
    assert len(extreme_pos) == n_corner_personas * 4
    print("=" * 70)
    print()

    print()
    print("=" * 70)
    print("Finding extreme left and right positions...")
    extreme_pos_left_right = find_left_right_extreme_positions(
        df, n=n_left_right_personas, use_center=use_center
    )
    assert len(extreme_pos_left_right) == n_left_right_personas * 2
    print("=" * 70)
    print()

    print()
    print("=" * 70)
    # Add suffix to filename if using center
    suffix = "_centered" if use_center else ""
    extreme_pos_path = (
        f"data/results/extreme_pos_personas/{MODEL}/extreme_pos_corners_{n_corner_personas}{suffix}.pkl"
    )
    os.makedirs(os.path.dirname(extreme_pos_path), exist_ok=True)
    print(f"Saving corner positions to: {extreme_pos_path}")
    with open(extreme_pos_path, "wb") as f:
        pickle.dump(extreme_pos, f)
    print("=" * 70)
    print()

    print()
    print("=" * 70)
    extreme_pos_left_right_path = (
        f"data/results/extreme_pos_personas/{MODEL}/extreme_pos_left_right_{n_left_right_personas}{suffix}.pkl"
    )
    print(f"Saving left and right positions to: {extreme_pos_left_right_path}")
    with open(extreme_pos_left_right_path, "wb") as f:
        pickle.dump(extreme_pos_left_right, f)
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=int,
        default=-1,
    )
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
        help="Use the center of the compass distribution as origin for diagonal calculations instead of (0,0)",
    )
    args = parser.parse_args()
    models = get_all_model_names()

    if args.model == -1:
        # loop on all models and plot
        for model_index, model in enumerate(models):
            print(f"Processing model {model_index + 1}/{len(models)}: {model}")
            process(
                model,
                n_corner_personas=args.n_corner_personas,
                n_left_right_personas=args.n_left_right_personas,
                use_center=args.use_center,
            )
    else:
        model = models[args.model]
        process(
            model,
            n_corner_personas=args.n_corner_personas,
            n_left_right_personas=args.n_left_right_personas,
            use_center=args.use_center,
        )

    print("Done!")


if __name__ == "__main__":
    main()