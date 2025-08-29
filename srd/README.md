# Compute No-Context/Context-based Prior Probability for Relation Deduction

## Preparation
- Unzip ```srd_data.zip```

## Step 1. Run K-Means (for Context-based SRD)
**Run the following command** (Revise the parameters if needed):
```bash
python srd/kmeans.py --train_feature_file srd_data/clip_image_feature.pth --num_clusters 25 --output_dir srd_data/output
```
where ```train_feature_file``` is the CLIP image feature for the images in the training set. 

**OUTPUT**:
- Trained K-Means model
- K-Means Predictions (for train set)


## Step 2. Run SRD (Statistical Relation Distillation)
**Run the following command** (Revise the parameters if needed): 
```bash
python srd/srd.py --output_dir srd_data/output --img_info_file srd_data/all_triplets.json --kmeans_model_path srd_data/output/kmeans_25.pkl --kmeans_train_prediction_path srd_data/output/kmeans_25_prediction.pkl --mapping_file srd_data/VG-SGG-dicts.json --relationship_file srd_data/relationships.json --split_train srd_data/train.json
```
where ```kmeans_model_path``` and ```kmeans_train_prediction_path``` are the files generated in *Step 1*. 

**OUTPUT**: File contains triple probability which will be used for the next step of *Relation Deduction* (data augmentation). There are two versions:
- No-Context: all the images in the train set are used 
- Context-based SRD: images of the corresponding cluster are used. Clusters are predicted using the trained K-Means model (Step 1)