import pandas as pd
import os
import glob
from tqdm import tqdm

results = pd.read_csv(
    "/data/DCASEfewshot/validate/eeefac1b1c19d0c7479c8d33ad588f833e9559aa/results/beats/data/results_20240418_063728/eval_out.csv",
    header=0,
    index_col=False,
)
set_type = "Validation_Set"
root_dir = "/data/DCASE/Development_Set_24"

all_csv_files = [
    file
    for path_dir, _, _ in os.walk(os.path.join(root_dir, set_type))
    for file in glob.glob(os.path.join(path_dir, "*.csv"))
]
for file in tqdm(all_csv_files):
    split_list = file.split("/")
    glob_cls_name = split_list[split_list.index(set_type) + 1]
    file_name = split_list[split_list.index(set_type) + 2]
    df = pd.read_csv(file, header=0, index_col=False)
    file_name_wav = file_name.replace("csv", "wav")

    # get average/min segment length for first five positives
    class_column = "Q"
    fifth_pos_row = df.index[df[class_column] == "POS"].tolist()[4]
    results.drop(
        results[results["Audiofilename"] == file_name_wav][
            results[results["Audiofilename"] == file_name_wav]["Starttime"]
            < df["Endtime"][fifth_pos_row]
        ].index,
        inplace=True,
    )

results.to_csv(
    # os.path.join(
    #     target_path,
    "eval_out_skipping_too_early_detections.csv",
    # ),
    index=False,
)
