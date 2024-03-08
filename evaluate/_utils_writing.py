import os
import librosa
import numpy as np
import pandas as pd

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
    resample=True,
    support_spectrograms=None,
    result_merged=None
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
            arr, fs_orig = librosa.load(f, sr=None, mono=True)
            break

    if not resample or target_fs > fs_orig:
        target_fs = fs_orig

    if resample:
        arr = librosa.resample(arr, orig_sr=fs_orig, target_sr=target_fs)
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
    merged_pred =np.zeros(len(arr))
    for  ind, row in result_merged.iterrows():
        merged_pred[int(row["Starttime"]*target_fs):int(row["Endtime"]*target_fs)] = 1




    # pad with zeros
    if len(arr) > len(gt_labels):
        gt_labels = np.pad(
            gt_labels, (0, len(arr) - len(gt_labels)), "constant", constant_values=(0,)
        )
        pred_labels = np.pad(
            pred_labels,
            (0, len(arr) - len(pred_labels)),
            "constant",
            constant_values=(0,),
        )
        distances_to_pos = np.pad(
            distances_to_pos,
            (0, len(arr) - len(distances_to_pos)),
            "constant",
            constant_values=(0,),
        )
        z_scores_pos = np.pad(
            z_scores_pos,
            (0, len(arr) - len(z_scores_pos)),
            "constant",
            constant_values=(0,),
        )
    else:
        arr = np.pad(
            arr,
            (0, len(z_scores_pos) - len(arr)),
            "constant",
            constant_values=(0,),
        )

    # Write the results
    result_wav = np.vstack(
        (arr, gt_labels, merged_pred, pred_labels , distances_to_pos / 10, z_scores_pos)
    )
    wavfile.write(output, target_fs, result_wav.T)

def plot_2_d_representation(prototypes,
                            z_pos_supports,
                            z_neg_supports,
                            q_embeddings,
                            labels,
                            output):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # Create a labels array for all points
    # Label for prototypes, positive supports, negative supports, and query embeddings respectively
    prototypes_labels = np.array([2] * prototypes.shape[0])  # Assuming 2 is not used in `gt_labels`
    pos_supports_labels = np.array([3] * z_pos_supports.shape[0])  # Assuming 3 is not used in `gt_labels`
    neg_supports_labels = np.array([4] * z_neg_supports.shape[0])  # Assuming 4 is not used in `gt_labels`
    q_embeddings = q_embeddings.detach().numpy()
    gt_labels = labels.detach().numpy()

    # Concatenate everything into one dataset
    feat = np.concatenate([prototypes.to("cpu").detach().numpy(), 
                           z_pos_supports.to("cpu").detach().numpy(), 
                           z_neg_supports.to("cpu").detach().numpy(), 
                           q_embeddings.to("cpu").detach().numpy()])
    all_labels = np.concatenate([prototypes_labels, pos_supports_labels, neg_supports_labels, gt_labels])

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30)
    features_2d = tsne.fit_transform(feat)

    # Plot
    plt.figure(figsize=(10, 8))
    # Define marker for each type of point
    markers = {2: "P", 3: "o", 4: "X"}  # P for prototypes, o for supports, X for negative supports

    for label in np.unique(all_labels):
        # Plot each class with its own color and marker
        idx = np.where(all_labels == label)
        if label in markers:  # Prototypes or supports
            plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=label, alpha=1.0, marker=markers[label], s=100)  # Larger size
        else:  # Query embeddings
            plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=label, alpha=0.5, s=50)  # Smaller size, more transparent

    plt.legend()
    plt.title('t-SNE visualization of embeddings, prototypes, and supports')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)

    # Save the figure
    plt.savefig(output, bbox_inches="tight")
    plt.show()