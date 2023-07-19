#!/bin/bash

download_and_unzip() {
    local file_url="$1"
    local target_folder="$2"

    # Extract the filename from the URL
    local filename=$(basename "${file_url%%\?*}")

    # Full path to the downloaded file
    local file_path="$target_folder/$filename"

    # Download the file
    wget -O "$file_path" "$file_url"

    # Check if the file is a zip file
    if file --mime-type "$file_path" | grep -q zip$; then
        echo "File is a zip. Unzipping..."
        unzip "$file_path" -d "$target_folder"
        echo "Deleting zip file..."
        rm "$file_path"
    fi
}

BASE_FOLDER=$1
TARGET_FOLDER=$BASE_FOLDER/DCASE
mkdir -p $TARGET_FOLDER

############################
# Download the BEATs model #
############################
MODEL_FOLDER=$BASE_FOLDER/BEATs
mkdir -p $MODEL_FOLDER
wget -O "$MODEL_FOLDER/BEATs_iter3_plus_AS2M.pt" "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D" 

#############################################################
# Download the development set (i.e. training + validation) #
#############################################################

# .csv
download_and_unzip "https://zenodo.org/record/6482837/files/DCASE2022_task5_Training_set_classes.csv?download=1" "$TARGET_FOLDER"
download_and_unzip "https://zenodo.org/record/6482837/files/DCASE2022_task5_Validation_set_classes.csv?download=1" "$TARGET_FOLDER"

# Annotations
download_and_unzip "https://zenodo.org/record/6482837/files/Development_Set_annotations.zip?download=1" "$TARGET_FOLDER"

# Acoustic data
download_and_unzip "https://zenodo.org/record/6482837/files/Development_Set.zip?download=1" "$TARGET_FOLDER"

###############################
# Download the evaluation set #
###############################
mkdir -p "$TARGET_FOLDER/Development_Set/Evaluation_Set"

download_and_unzip "https://zenodo.org/record/7879692/files/Annotations_only.zip?download=1" "$TARGET_FOLDER"
mv "$TARGET_FOLDER/Annotations_only" "$TARGET_FOLDER/Development_Set_annotations/Evaluation_Set"

# Acoustic data
for i in {1..3}
do
    download_and_unzip "https://zenodo.org/record/7879692/files/eval_$i.zip?download=1" "$TARGET_FOLDER/Development_Set/Evaluation_Set"
done


