'''
API to be used for scene graph generation 
'''
import os 
import pickle as pkl 
import json
from maskrcnn_benchmark.utils.miscellaneous import loadf


class ContextKB:
    def __init__(self, kmeans_model_path, cluster_to_knowledge_file='kg/data/cb_prior_knowledge/clusters/cluster2knowledge.json', 
                 kmeans_pred_file='kg/data/cb_prior_knowledge/kmeans/kmeans_25_prediction_combined_img2cluster.pkl'):
        print("Loading k-means model:", kmeans_model_path)
        self.kmeans = pkl.load(open(kmeans_model_path, 'rb'))
        print("Loading pre-clustered images:", kmeans_pred_file)
        self.img2cluster = pkl.load(open(kmeans_pred_file, 'rb'))
        if os.path.isfile(cluster_to_knowledge_file): 
            print("Loading context-based prior knowledge: ", cluster_to_knowledge_file)
            self.cluster2info = loadf(cluster_to_knowledge_file)
            if 'cluster2pair_prob' in self.cluster2info:
                self.cluster2pair_prob = self.cluster2info['cluster2pair_prob']  # {cluster: {subject: {object: score}}}
                self.cluster2pred_prob = self.cluster2info['cluster2pred_prob']  # {cluster: {subject: {object: verb}}}
        else:
            print("[FAILED] Invalid precomputed context-based knowledge file (cluster_to_knowledge_file: {})".format(cluster_to_knowledge_file))
    def get_prior_knowledge(self, img_id):
        cluster = self.img2cluster.get(img_id, -1)
        if cluster < 0: 
            return None, None 
        cluster = str(cluster)
        return self.cluster2pair_prob[cluster], self.cluster2pred_prob[cluster]
    
    
if __name__ == '__main__':
    ckb = ContextKB(kmeans_model_path='kg/data/cb_prior_knowledge/kmeans/kmeans_25.pkl', 
                    kmeans_pred_file='kg/data/cb_prior_knowledge/kmeans/kmeans_25_prediction_combined_img2cluster.pkl', 
                    cluster_to_knowledge_file='kg/data/cb_prior_knowledge/clusters/cluster2knowledge.json')
    pair_prob, pred_prob = ckb.get_prior_knowledge(img_id=2415821)
    pair_prob['shirt']
    pred_prob['shirt']
