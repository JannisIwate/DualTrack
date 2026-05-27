from argparse import ArgumentParser
import os
import csv


def main():
    p = ArgumentParser()

    p.add_argument(
        "--data_split",
        action="store_true",
        help="Set if frames and transforms are stored separately.",
    )

    p.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to the output dir.",
    )

    # Argument for input_tforms (only valid with --data_split)
    p.add_argument(
        "--input_tforms",
        "-it",
        help="Transforms base path (required when --data_split is set).",
    )

    # Arguments for when --data_split is set
    p.add_argument(
        "--input_frames",
        "-if",
        help="Frames base path (required when --data_split is set).",
    )

    # Arguments for when --data_split is NOT set
    p.add_argument(
        "--input",
        "-i",
        help="Base path containing combined .h5 files (required when --data_split is NOT set).",
    )

    args = p.parse_args()

    # Validate argument combinations
    if args.data_split:
        if not args.input_frames or not args.input_tforms:
            raise ValueError(
                "When using --data_split, both --input_frames and --input_tforms are required."
            )
    else:
        if args.input_frames or args.input_tforms:
            raise ValueError(
                "--input_frames/_tforms can only be used with --data_split."
            )

    output_csv = os.path.abspath(args.output) + "/conversion_sweeps.csv"

    data = []
    sweep_id_counter = 0

    # -------------------------------------------------------------------------
    # SPLIT DATASET (frames + tforms separately)
    # -------------------------------------------------------------------------

    if args.data_split:

        frames_base_path = os.path.abspath(args.input_frames)
        tforms_base_path = os.path.abspath(args.input_tforms)

        # loop through top-level folders
        for top_folder in os.listdir(frames_base_path):

            frames_top_path = os.path.join(frames_base_path, top_folder)
            tforms_top_path = os.path.join(tforms_base_path, top_folder)

            if not os.path.isdir(frames_top_path):
                continue

            # loop through files
            for file in os.listdir(frames_top_path):

                if not file.endswith(".h5"):
                    continue

                frames_file_path = os.path.join(frames_top_path, file)
                tforms_file_path = os.path.join(tforms_top_path, file)

                data.append(
                    {
                        "sweep_id": f"sweep_{sweep_id_counter:05d}",
                        "raw_tus_rec_frames_path": frames_file_path,
                        "raw_tus_rec_tforms_path": tforms_file_path,
                        "split": "train",
                    }
                )

                sweep_id_counter += 1

    # -------------------------------------------------------------------------
    # COMBINED DATASET (frames + tforms in same file)
    # -------------------------------------------------------------------------

    else:

        base_path = os.path.abspath(args.input)

        for top_folder in os.listdir(base_path):

            top_folder_path = os.path.join(base_path, top_folder)

            if not os.path.isdir(top_folder_path):
                continue

            for file in os.listdir(top_folder_path):

                if not file.endswith(".h5"):
                    continue

                file_path = os.path.join(top_folder_path, file)

                data.append(
                    {
                        "sweep_id": f"sweep_{sweep_id_counter:05d}",
                        "raw_tus_rec_sweep_path": file_path,
                        "split": "train",
                    }
                )

                sweep_id_counter += 1

    # -------------------------------------------------------------------------
    # WRITE CSV
    # -------------------------------------------------------------------------

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as csvfile:

        if args.data_split:
            fieldnames = [
                "sweep_id",
                "raw_tus_rec_frames_path",
                "raw_tus_rec_tforms_path",
                "split",
            ]
        else:
            fieldnames = [
                "sweep_id",
                "raw_tus_rec_sweep_path",
                "split",
            ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data)

    print(f"CSV created: {output_csv}")
    print(f"Total sweeps: {len(data)}")


if __name__ == "__main__":
    main()