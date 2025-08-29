
import torch
import json 
from sklearn.cluster import KMeans
import numpy as np
import pickle as pkl
# import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import os 
import time
import argparse


def mkdirs(idir):
    if not os.path.isdir(idir):
        os.makedirs(idir)
        
        
def get_id(img, prefix='datasets/vg/VG_100K/'):
    if not img.startswith(prefix):
        print(img)
        return None 
    return int(img[len(prefix):-4])


def get_features(features_data, prefix='datasets/vg/VG_100K/'):
    img_ids = list(features_data.keys())
    if isinstance(img_ids[0], str) and img_ids[0].startswith(prefix):
        img_ids = [get_id(img, prefix=prefix) for img in features_data.keys()]
    features = list(features_data.values())
    features = torch.stack(features).numpy()
    return img_ids, features 


def run_kmeans(ifile, k=25, random_state=23):
    odir = args.output_dir
    mkdirs(odir)
    dt = torch.load(ifile, map_location=torch.device('cpu'))
    img_ids, features_nd = get_features(dt)  # get features for all the images in the features file 
    # print(f"Number of images: {len(img_ids)}")
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=random_state).fit(features_nd)  # run k-means 
    model_ofile = os.path.join(odir, "kmeans_{}.pkl".format(k))
    pkl.dump(kmeans, open(model_ofile, 'wb'))
    print("K-Means Model (K={}) is saved to {}".format(k, model_ofile))
    pred_ofile = os.path.join(odir, "kmeans_{}_prediction.pkl".format(k))
    run_kmeans_predict(kmeans, features_nd, pred_ofile, img_ids)
    print("K-Means prediction is saved to {}".format(pred_ofile))
    return kmeans, img_ids, model_ofile, pred_ofile


def run_kmeans_predict(kmeans, features_nd, ofile, img_ids):
    y = kmeans.predict(features_nd)
    pkl.dump({'img_ids': img_ids, 'prediction': y}, open(ofile, 'wb'))
    print("Saved prediction to {}".format(ofile))


def run_kmeans_inference(kmean_file, feature_file):
    ofile=os.path.join(args.output_dir, f'{os.path.basename(feature_file)[:-4]}_kmeans_prediction.pkl')
    kmeans = pkl.load(open(kmean_file, 'rb'))
    dt = torch.load(feature_file, map_location=torch.device('cpu'))
    img_ids, features_nd = get_features(dt)  # get features for all the images in the features file 
    run_kmeans_predict(kmeans, features_nd, ofile, img_ids)
    return ofile 

if __name__ == "__main__":
    parser = argparse.ArgumentParser("K-Means")
    parser.add_argument('--train_feature_file', default='srd_data/clip_image_feature.pth')
    # parser.add_argument('--val_feature_file', default='srd_data/vg/clip_image_feature_val.pth')
    # parser.add_argument('--test_feature_file', default='srd_data/vg/clip_image_feature_test.pth')
    parser.add_argument('--num_clusters', help='Number of clusters (K)', default=25, type=int)
    parser.add_argument('--random_state', help='Random state', default=23, type=int)
    parser.add_argument('--output_dir', default='srd_data/output')
    
    args = parser.parse_args()
    train = args.train_feature_file
    # val = args.val_feature_file
    # test = args.test_feature_file
    output_dir = args.output_dir
    kmeans, img_ids, model_path, train_pred = run_kmeans(ifile=train, k=args.num_clusters, random_state=23)
    
    # 2. Predict for val and test 
    # val_pred = run_kmeans_inference(kmean_file=model_path, feature_file=val)
    # test_pred = run_kmeans_inference(kmean_file=model_path, feature_file=test)