import os
import sys
import h5py
import numpy as np
from tqdm import tqdm
from itertools import islice

sys.path.append(os.getcwd())
sys.path.append("/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/pgo")
[sys.path.append(i) for i in [".", ".."]]

from pose_graph_optimization.graph import *
from pose_graph_optimization.error_metrics import *
from pose_graph_optimization.utils import *
from src.utils.pose import get_global_and_relative_gt_trackings
from src.utils.pose import get_drift_metrics, get_ddf_metrics, get_global_and_relative_gt_trackings, matrix_to_pose_vector


def main():

    # paths
    input_pred_path = "/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/experiment/dualtrack_24/tusrec_24_val/validation_run/scans"
    input_gt_path = "/mnt/c/Users/Jannis/Documents/Thesis_Prima/DualTrack/DualTrack_auxiliary/validation_data_tusrec24_converted"

    # data
    data = os.listdir(input_pred_path)
    nr_of_scans = None
    start = 1
    data = islice(data, start-1, start+nr_of_scans-1) if nr_of_scans is not None else data

    pred_inbetween_all = None
    gt_inbetween_all = None

    for i, el in enumerate(tqdm(data, desc="Working", total=nr_of_scans)):

        ## load data
        # load file
        sweep_path = os.path.join(input_pred_path, el, "export.h5")
        sweep_name = f"sweep_{i}"

        if not os.path.isfile(sweep_path):
            continue

        with h5py.File(sweep_path, "r") as f:

            nr_of_frames= 2
            nr_of_frames = len(f["images"])
            
            # load scan data
            pred_acc = np.array(f["pred_tracking_glob"][:nr_of_frames]) # starts with identity, normalized acc world coords
            pred_inbetween = np.array(f["pred_tracking_loc"][:nr_of_frames])

            if pred_inbetween_all is None:
                pred_inbetween_all = pred_inbetween
            else:
                pred_inbetween_all = np.concatenate((pred_inbetween_all, pred_inbetween), axis=0)

            gt_file = os.path.join(input_gt_path, f"{el}.h5")

            with h5py.File(gt_file, "r") as f_gt:
                gt = np.array(f_gt["tracking"][:nr_of_frames]) # acc gt poses in arbitrary world coords
                gt_acc, gt_inbetween = get_global_and_relative_gt_trackings(gt) # get normalized acc gt (first pose is identity) and relative gt
                # first transform of gt_acc is identity (or rather almost due to numerics)
                # first transform of gt_inbetween is identity as first frame is first frame
                # gt_inbetween is Ti->j, forward
            
            if gt_inbetween_all is None:
                gt_inbetween_all = gt_inbetween
            else:
                gt_inbetween_all = np.concatenate((gt_inbetween_all, gt_inbetween), axis=0)
        
    plot_pose_differences_j(pred_inbetween_all, gt_inbetween_all, title="Error vs GT")
    # plt.show()


def plot_pose_differences_j(pred, gt, title=None, ax=None):

    pred_tracking = np.stack([matrix_to_pose_vector(matrix) for matrix in pred])
    gt_tracking = np.stack([matrix_to_pose_vector(matrix) for matrix in gt])

    errors_abs = np.abs(pred_tracking - gt_tracking)
    errors_real = (pred_tracking - gt_tracking) # errors are in mm and degrees
    errors_abs_mean = errors_abs.mean(0)
    errors_real_mean = errors_real.mean(0)
    mean_error_array = np.cumsum(errors_real, axis=0) / np.arange(1, len(errors_real) + 1)[:, None]

    # print(np.abs(gt_tracking).mean(0))
    # print(errors_abs_mean)

    for i in range(errors_real.shape[1]):
        # plt.figure()
        # plt.hist(errors_real[:, i], bins=60)
        # plt.title(f"Column {i}")
        # plt.xlabel("Value")
        # plt.ylabel("Count")


        lower = np.percentile(errors_real[:, i], 0)
        upper = np.percentile(errors_real[:, i], 100)

        errors_real_filtered = errors_real[:, i][(errors_real[:, i] >= lower) & (errors_real[:, i] <= upper)]

        import scipy.stats as stats
        import matplotlib.pyplot as plt
        stats.probplot(errors_real_filtered, dist="norm", plot=plt)
        # sortiere Punkte, weise jedem Punkt ein Quantil zu, berechne Idealverteilung anhand von Standardabweichung und Mittelwert, plotte Idealquantile und tatsaechliche Werte

        std_population = np.std(errors_real[:, i])
        print(std_population)
        std_gt = np.std(gt_tracking[:, i])
        print(std_gt)
        print("\n")
        # -> standard deviation for errors
        # x: 0.07656708383374906
        # y: 0.02681774961755883
        # z: 0.09021679400248785
        # pitch: 0.04206508373865844
        # yaw: 0.033058318883716194
        # roll: 0.03825069374999717

    # -> errors are basically gaussian (only really small and really big errors are off)

    # plt.show()

    if ax is None: 
        fig, ax = plt.subplots(2, 3)
    else: 
        fig = plt.gcf()

    for i in range(6):
        ax_ = ax.flatten()[i]
        tags = ["x", "y", "z", "pitch", "roll", "yaw"]

        # ax_.plot(pred_tracking[:, i], label="pred", alpha=0.8, color="orange")
        # ax_.plot(np.abs(gt_tracking)[:, i], label="gt", alpha=0.8, color="blue")
        # ax_.plot(gt_tracking[:, i], label="gt", alpha=0.8, color="blue")
        # ax_.plot(errors_abs[:, i], label="error to pred", alpha=0.8, color="orange")
        #ax_.plot(mean_error_array[:, i], label="error mean through time", alpha=0.8, color="orange")
        # ax_.set_title(f"mae={errors_abs_mean[i]:.2f}, {errors_real_mean[i]:.2f}")

        # -> Fehler sind im Prinzip mittelwertfrei
        # -> Groessere Fehler bei groesseren Werten
        # -> Groesste Fehler bei y und roll (ergibt Sinn, da y Dimension kleiner ist als x Dimension -> weniger Info bei y, mehr Fehler bei x roll)
        # -> Generell sind Winkelfehler viel groesser als T Fehler im Vergleich zu Werten (ergibt Sinn, da Translation recht eindeutig ist)
        # TODO: Ueberlegen, wie man diese Erkenntnisse nutzt
        # TODO: Plotte das hier alles fuer PGO und IR Schaetzungen, vorher speichern

        if i <= 2:
            ax_.set_ylabel(f"{tags[i]} (mm)")
        else:
            ax_.set_ylabel(f"{tags[i]} (°)")
        ax_.set_xlabel(f"timestep")

        if i == 5:
            ax_.legend()
    title="GT vs Pred"
    fig.canvas.manager.set_window_title(title)
    fig.tight_layout()
    
    return fig


if __name__ == "__main__":
    main()