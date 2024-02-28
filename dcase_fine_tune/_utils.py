import os
import librosa
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

def write_results(predicted_labels, begins, ends):
    df_out = pd.DataFrame(
        {
            "Starttime": begins,
            "Endtime": ends,
            "PredLabels": predicted_labels,
        }
    )

    return df_out

def write_wav(
    files,
    cfg,
    gt_labels,
    pred_labels,
    distances_to_pos,
    z_scores_pos,
    target_fs=16000,
    target_path=None,
    frame_shift=1,
    support_spectrograms=None
):
    from scipy.io import wavfile

    # Some path management
    filename = (
        os.path.basename(support_spectrograms).split("data_")[1].split(".")[0] + ".wav"
    )
    # Return the final product
    output = os.path.join(target_path, filename)

    # Find the filepath for the file being analysed
    for f in files:
        if os.path.basename(f) == filename:
            print(os.path.basename(f))
            print(filename)
            arr, _ = librosa.load(f, sr=target_fs, mono=True)
            break

    # Expand the dimensions
    gt_labels = np.repeat(
        np.squeeze(gt_labels, axis=1).T,
        int(
            cfg["data"]["tensor_length"]
            * cfg["data"]["overlap"]
            * target_fs
            * frame_shift
            / 1000
        ),
    )
    pred_labels = np.repeat(
        pred_labels.T,
        int(
            cfg["data"]["tensor_length"]
            * cfg["data"]["overlap"]
            * target_fs
            * frame_shift
            / 1000
        ),
    )
    distances_to_pos = np.repeat(
        distances_to_pos.T,
        int(
            cfg["data"]["tensor_length"]
            * cfg["data"]["overlap"]
            * target_fs
            * frame_shift
            / 1000
        ),
    )
    z_scores_pos = np.repeat(
        z_scores_pos.T,
        int(
            cfg["data"]["tensor_length"]
            * cfg["data"]["overlap"]
            * target_fs
            * frame_shift
            / 1000
        ),
    )

    # pad with zeros
    gt_labels = np.pad(
        gt_labels, (0, len(gt_labels) - len(arr)), "constant", constant_values=(0,)
    )
    pred_labels = np.pad(
        pred_labels, (0, len(pred_labels) - len(arr) ), "constant", constant_values=(0,)
    )
    distances_to_pos = np.pad(
        distances_to_pos,
        (0, len(distances_to_pos) - len(arr)),
        "constant",
        constant_values=(0,),
    )
    z_scores_pos = np.pad(
        z_scores_pos,
      (0, len(z_scores_pos) - len(arr)),
        "constant",
        constant_values=(0,),
    )

    # Write the results
    result_wav = np.vstack(
        (arr, gt_labels, pred_labels, distances_to_pos / 10, z_scores_pos)
    )
    wavfile.write(output, target_fs, result_wav.T)

def merge_preds(df, tolerence, tensor_length):
    df["group"] = (
        df["Starttime"] > (df["Endtime"] + tolerence * tensor_length).shift().cummax()
    ).cumsum()
    result = df.groupby("group").agg({"Starttime": "min", "Endtime": "max"})
    return result

def to_dataframe(features, labels):
    """Load the saved array and map the features and labels into a single dataframe"""
    input_features = np.load(features)
    labels = np.load(labels)
    list_input_features = [input_features[key] for key in input_features.files]
    df = pd.DataFrame({"feature": list_input_features, "category": labels})
    return df

def compute_scores(predicted_labels, gt_labels):
    acc = accuracy_score(gt_labels, predicted_labels)
    recall = recall_score(gt_labels, predicted_labels)
    f1score = f1_score(gt_labels, predicted_labels)
    precision = precision_score(gt_labels, predicted_labels)
    print(f"Accurracy: {acc}")
    print(f"Recall: {recall}")
    print(f"precision: {precision}")
    print(f"F1 score: {f1score}")
    return acc, recall, precision, f1score

def construct_path(base_dir, status, hash_dir_name, file_type, file_pattern):
    return os.path.join(base_dir, status, hash_dir_name, "audio", f"{file_type}.{file_pattern}")