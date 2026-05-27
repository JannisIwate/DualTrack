from argparse import ArgumentParser
import os
import csv


def main():
    p = ArgumentParser()

    p.add_argument(
        "--input",
        "-i",
        required=True,
        help="Base path containing combined .h5 files in DualTrack format.",
    )

    p.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to the output dir.",
    )

    args = p.parse_args()

    base_path = os.path.abspath(args.input)
    output_csv = os.path.abspath(args.output) + "/sweeps.csv"

    # -------------------------------------------------------------------------
    # DATA COLLECTION
    # -------------------------------------------------------------------------

    data = []
    sweep_id_counter = 0

    # combined frames+tforms in same h5 file

    for file in os.listdir(base_path):

        if not file.endswith(".h5"):
            continue

        file_path = os.path.join(base_path, file)

        data.append({
            "index": sweep_id_counter,
            "sweep_id": f"sweep_{sweep_id_counter:05d}",
            "processed_sweep_path": file_path,
            "split": "val"
        })
        sweep_id_counter += 1

    # -------------------------------------------------------------------------
    # WRITE CSV
    # -------------------------------------------------------------------------

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = [
            "index",
            "sweep_id",
            "processed_sweep_path",
            "split"
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data)

    print(f"CSV created: {output_csv}")
    print(f"Total sweeps: {len(data)}")


if __name__ == "__main__":
    main()