import os 
import sys 
from os.path import dirname
import json 

PROJECT_DIR = dirname(dirname(dirname(os.path.abspath(__file__))))
# print(PROJECT_DIR)
sys.path.append(PROJECT_DIR)

# from kg.preprocessing.context_based.kmeans_v2 import KMeans
# from kg.preprocessing.context_based.knowledge_extractor import KnowledgeExtractor
import pickle as pkl 
# from kg.preprocessing.context_based import postprocess_prior_knowledge as pp 
from scipy.sparse import csr_matrix
import torch 
from torch.nn.functional import normalize
from tqdm import tqdm 
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import sklearn.metrics.pairwise as pw
import torch.nn.functional as F
import argparse


def load_list_as_dict(ifile, key):
    dt = json.load(open(ifile))
    k2v = {}
    for d in dt: 
        k2v[d[key]] = d
    return k2v 

def get_sims_with_threshold(sims, threshold=0.7):
    tmp = sims.clone()
    if threshold is None: 
        return tmp
    tmp[tmp < threshold] = 0
    return tmp 

def load_idx_mapping(ifile):
        data = json.load(open(ifile))
        label_to_idx = data['label_to_idx']
        idx_to_label = data['idx_to_label']
        predicate_to_idx = data['predicate_to_idx']
        idx_to_predicate = data['idx_to_predicate']
        return label_to_idx, idx_to_label, predicate_to_idx, idx_to_predicate
    
def compute_embeddings(keywords, emb_class='sentence_transformer', save_embeddings=False, odir='kg/data/kgdata/objs_sim/', verbose=False):
    if verbose:
        print("#keywords: {}".format(len(keywords)))
    # compute embeddings 
    if emb_class == "sentence_transformer":
        if verbose: 
            print("Using sentence_transformer: paraphrase-mpnet-base-v2")
        model = SentenceTransformer('paraphrase-mpnet-base-v2')
        embds = model.encode(keywords, convert_to_tensor=True, show_progress_bar=True)  # compute embeddings 
        embds = embds.cpu()
    # elif emb_class == 'word2vec':
    #     print("Using {}: word2vec-google-news-300".format(emb_class))
    #     model = api.load('word2vec-google-news-300')
    #     embds = []
    #     for w in tqdm(keywords): 
    #         embds.append(model[w])
    #     embds = np.stack(embds, axis=0)
    else:
        embds = None 
    if embds is None: 
        print("Cannot compute word embeddings")
        return None
    elif save_embeddings:
        eofile = os.path.join(odir, "{}_embeddings.pkl".format(emb_class))
        pkl.dump(embds, open(eofile, 'wb'))
        print("Saved embeddings in", eofile)
    if verbose:
        print("Embeddings shape: {}".format(embds.shape))
    return embds

class KnowledgeExtractor:
    def __init__(self, relationship_file='kg/data/relationships.json', 
                 mapping_file='datasets/vg/VG-SGG-dicts.json', 
                 img_info_file='datasets/vg/all_triplets.json', 
                 embedding_class='sentence_transformer', simple=False):
        print("Initializing KnowledgeExtractor")
        self.lemmatizer = WordNetLemmatizer()
        print("Loading images info")
        self.img2info = load_list_as_dict(relationship_file, key='image_id')  # information
        print("Loading mappings") 
        self.n_size = 150  # number of selected objects 
        self.label_to_idx, self.idx_to_label, self.pred_to_idx, self.idx_to_pred = load_idx_mapping(mapping_file)
        self.selected_objs = list(self.idx_to_label.values())  # for computing cosine similarity 
        self.selected_preds = list(self.idx_to_pred.values())  # for computing cosine similarity 
        self.embedding_class = embedding_class
        self.sel_objs_embs, self.sel_preds_emds = None, None 

    def extract_knowledge(self, img_ids, odir, save_logs=True, self_distillation=False, with_distillation=True, min_triple_freq=2, 
                          lemmatize=True, verbose=False, obj_threshold=0.7, pred_threshold=0.7, min_pair_frequency=5, 
                          rel_prob_threshold = 0.1, rel_freq_threshold = 3, int_files_dir=None, 
                          pred_sim_file=None, obj_sim_file=None):
        '''
        @pred_sim_file: will use when provided. This is for data-driven predicate similarity 
        @aggregate_rel: whether to perform aggregation, i.e., combining frequency of similar objs/predicates
        @self_distillation: only use the selected objects and predicates 
        '''
        print("-----------------------------------------------------------------------")
        print("Running Knowledge Extraction. With distillation?: {}".format(with_distillation))
        mkdirs(odir)
        sfile = os.path.join(odir, 'settings.json')
        settings = {'aggregate_rel': with_distillation, 'min_triple_freq': min_triple_freq, 
                    'lemmatize': lemmatize, 'obj_threshold': obj_threshold, 'pred_threshold': pred_threshold, 
                    'min_pair_frequency': min_pair_frequency, 'self_distillation': self_distillation, 'pred_sim_file': pred_sim_file, 
                    'obj_sim_file': obj_sim_file}
        json.dump(settings, open(sfile, 'w'))
        # 1. get all triples 
        triple2count, all_objs_otf, all_preds_otf = self.extract_all_triples(img_ids, lemmatize, min_triple_freq, verbose)
        if save_logs:
            # pkl.dump(all_objs, open(os.path.join(odir, 'all_objs.pkl'), 'wb'))
            # pkl.dump(all_preds, open(os.path.join(odir, 'all_preds.pkl'), 'wb'))
            pkl.dump(triple2count, open(os.path.join(odir, 'triple2count.pkl'), 'wb'))
        # 2. aggregation 
        if with_distillation:
            pred_sim, all_preds_pc = self.prepare_precomputed_sim(pred_sim_file, self_distillation=self_distillation)
            obj_sim, all_objs_pc = self.prepare_precomputed_sim(obj_sim_file, self_distillation=self_distillation)
            if self_distillation:
                freq_matrix = self.aggregate_frequency(triples=triple2count, self_distillation=True, 
                                                   obj_threshold=obj_threshold, pred_threshold=pred_threshold, 
                                                   verbose=verbose, odir=odir, int_files_dir=int_files_dir, pred_sim=pred_sim, obj_sim=obj_sim)
            else:
                if all_objs_pc is None: 
                    all_objs_pc = all_objs_otf
                if all_preds_pc is None: 
                    all_preds_pc = all_preds_otf
                freq_matrix = self.aggregate_frequency(triples=triple2count, all_objs=all_objs_pc, all_preds=all_preds_pc,
                                                    obj_threshold=obj_threshold, pred_threshold=pred_threshold, 
                                                    verbose=verbose, odir=odir, int_files_dir=int_files_dir, pred_sim=pred_sim, obj_sim=obj_sim)
        else:
            freq_matrix = self.get_non_aggregated_matrix(triple2count=triple2count) 
        # if save_logs:
        pkl.dump(freq_matrix, open(os.path.join(odir, 'freq_matrix.pkl'), 'wb'))
        # 3. create human readable prior knowledge 
        subject2pair_prob = self.prepare_pair_prob(freq_matrix, min_freq=min_pair_frequency)
        # pkl.dump(subject2pair_prob, open(os.path.join(odir, 'subject2pair_prob.pkl'), 'wb'))
        json.dump(subject2pair_prob, open(os.path.join(odir, 'subject2pair_prob.json'), 'w'))
        # 4. create subject-object to relation distribution 
        so2pred = prepare_top_pred_for_subj_obj(freq_matrix, idx_to_label=self.idx_to_label, idx_to_pred=self.idx_to_pred, rel_prob_threshold=rel_prob_threshold, rel_freq_threshold=rel_freq_threshold)
        # pkl.dump(so2pred, open(os.path.join(odir, 'subj_obj_to_top_pred.pkl'), 'wb'))
        json.dump(so2pred, open(os.path.join(odir, 'subj_obj_to_top_pred.json'), 'w'))
        return subject2pair_prob, so2pred, freq_matrix

    def prepare_precomputed_sim(self, ifile, self_distillation=False):
        if ifile is None or not os.path.isfile(ifile):  # invalid file 
            return None, None 
        dt = pkl.load(open(ifile, 'rb'))  # dt = {'all_preds': all_preds, 'sel_preds': ke.selected_preds, 'sim': b}
        if isinstance(dt, dict):
            all_items_pc = dt['all']
            sim = dt['sim']
        else:
            sim = dt  # should not go into this 
            all_items_pc = None   # all_preds for precomputed
        if self_distillation:
            if all_items_pc is None: 
                print("Need `all_items' information for self-distillation")
                return None, None 
            # only choose the rows of selected predicates 
            tmp = []
            for p in self.selected_preds:
                tmp.append(sim[all_items_pc.index(p), :])
            sim = torch.stack(tmp)
        return sim.to(torch.float32), all_items_pc

    def get_non_aggregated_matrix(self, triple2count):
        '''
        @output: padded matrix 
        '''
        ps = len(self.selected_preds)
        ns = len(self.selected_objs)
        non_agg_freq = torch.zeros((ps+1, ns+1, ns+1)) # 51 x 151 x 151
        for t, c in tqdm(triple2count.items(), total=len(triple2count)):
            s, p, o = t
            pidx = self.pred_to_idx.get(p, None)
            sidx = self.label_to_idx.get(s, None)
            oidx = self.label_to_idx.get(o, None)
            if pidx is not None and sidx is not None and oidx is not None: 
                # if oidx == 0: 
                #     print("oidx == 0")
                #     print(t)
                #     print(c)
                #     input()
                non_agg_freq[pidx, sidx, oidx] = c
        return non_agg_freq

    def prepare_pair_prob(self, prior_knowledge, min_freq=5):
        if prior_knowledge is None: 
            return None 
        prior_objs = torch.sum(prior_knowledge, dim=0)  # [151 x 151]
        s2op = {}  # {subj: {obj: prob}}
        for sidx in range(1, 151):
            subj = self.idx_to_label[str(sidx)]
            if subj not in s2op:
                s2op[subj] = {}
            objs_freq = prior_objs[sidx]
            obj_norm = normalize(objs_freq, p=1, dim=0)
            non_zero_indices = torch.nonzero(obj_norm)
            for oidx in non_zero_indices:
                oidx = oidx.item()
                # print("subj: {} obj: {}".format(sidx, oidx))
                obj = self.idx_to_label[str(oidx)]
                freq = objs_freq[oidx].item()
                if freq >= min_freq:
                    # s2op[subj][obj] = (obj_norm[oidx].item(), freq)
                    s2op[subj][obj] = obj_norm[oidx].item()
        return s2op

    def extract_all_triples(self, img_ids, lemmatize=False, min_triple_freq=2, verbose=False):
        if verbose: 
            print("----------------------------------")
            print("Extract all triples. #reviews: {}".format(len(img_ids)))
            print("----------------------------------")
        t2c = self.count_triples(img_ids, lemmatize=lemmatize)  # triple: frequency
        # filter triples by frequency 
        if verbose: 
            print("#triples: {}".format(len(t2c)))
        if min_triple_freq > 1: 
            t2c = self.filter_triples_by_freq(t2c, min_freq=min_triple_freq)
            if verbose: 
                print("#filtered triples (min freq: {}): {}".format(min_triple_freq, len(t2c)))
        # get all objects and predicates 
        if verbose:
            print("Prepare ALL objects and predicates")  
        all_objs = self.get_all_terms(t2c, indices=[0,2])  # 42016
        all_preds = self.get_all_terms(t2c, indices=[1])  # 22481
        return t2c, all_objs, all_preds

    def get_all_terms(self, t2c, indices=[0,2]):
        tps = list(t2c.keys())
        tmp = []
        for idx in indices:
            l = [a[idx] for a in tps]
            tmp += l
        return list(set(tmp))

    def compute_cosine_sims(self, all_objs, all_preds=None, obj_threshold=0.7, pred_threshold=0.7):
        print("Computing cosine similarity between all objs and selected objs")
        all_objs_embs = compute_embeddings(keywords=all_objs, emb_class=self.embedding_class)
        if self.sel_objs_embs is None: 
            self.sel_objs_embs = compute_embeddings(keywords=self.selected_objs, emb_class=self.embedding_class)
        objs_sim = compute_cosine_similarity(all_objs_embs, self.sel_objs_embs)        
        objs_sim = torch.from_numpy(objs_sim)  # convert to tensor 
        objs_sim_07 = get_sims_with_threshold(sims=objs_sim, threshold=obj_threshold)
        if all_preds is not None: 
            print("Computing cosine similarity between all preds and selected preds")
            all_preds_emds = compute_embeddings(keywords=all_preds, emb_class=self.embedding_class)
            if self.sel_preds_emds is None: 
                self.sel_preds_emds = compute_embeddings(keywords=self.selected_preds, emb_class=self.embedding_class)
            preds_sim = compute_cosine_similarity(all_preds_emds, self.sel_preds_emds)
            preds_sim = torch.from_numpy(preds_sim)
            preds_sim_07 = get_sims_with_threshold(sims=preds_sim, threshold=pred_threshold)
        else:
            preds_sim_07 = None 
        return objs_sim_07, preds_sim_07

    def aggregate_frequency(self, triples, all_objs=None, all_preds=None, self_distillation=False, obj_threshold=0.7, 
                            pred_threshold=0.7, verbose=False, odir=None, save_logs=True, int_files_dir=None, pred_sim=None, obj_sim=None):
        if verbose:
            print("----------------------------------")
            print("Aggregate frequency")
            print("----------------------------------")
        if self_distillation:
            print("SELF DISTILLATION")
            all_objs = self.selected_objs
            all_preds = self.selected_preds
            pred_list = all_preds
        else:
            if all_objs is None or all_preds is None: 
                print("all_objs and all_preds must not be None when self_distillation==False")
                return None 
            pred_list = None 
        # compute cosine similarity 
        if obj_sim is None:  # for now assuming if obj sim is not None then pred sim is also provided (not none)
            if pred_sim is None: 
                obj_sim, pred_sim = self.compute_cosine_sims(all_objs, all_preds, obj_threshold, pred_threshold)
                if odir is not None and save_logs: 
                    obj_ofile = os.path.join(odir, 'objs_sim_{}-with_item_list.pkl'.format(obj_threshold))
                    pred_ofile = os.path.join(odir, 'preds_sim_{}-with_item_list.pkl'.format(obj_threshold))
                    if not os.path.isfile(obj_ofile):
                        pkl.dump({"all": all_objs, 'sim': obj_sim}, open(obj_ofile, 'wb'))
                        pkl.dump({"all": all_preds, 'sim': pred_sim}, open(pred_ofile, 'wb'))
            else:
                obj_sim, _ = self.compute_cosine_sims(all_objs, all_preds=None, obj_threshold=obj_threshold, pred_threshold=pred_threshold)
                if odir is not None and save_logs: 
                    obj_ofile = os.path.join(odir, 'objs_sim_{}.pkl'.format(obj_threshold))
                    if not os.path.isfile(obj_ofile):
                        pkl.dump(obj_sim, open(obj_ofile, 'wb'))
        # prepare sparse matrix 
        pred2matrix = self.prepare_sparse_matrix_for_pred(triples, all_objs, pred_list=pred_list)  # all predicates 
        if int_files_dir is not None: 
            mkdirs(int_files_dir)
            pkl.dump(pred2matrix, open(os.path.join(int_files_dir, 'pred2matrix.pkl'), 'wb'))
        # for each predicate: multiply the sparse matrix (MxM) with (MxN)
        if verbose: 
            print("Compute FREQUENCY MATRIX (MxN) = FREQ MATRIX (MxM) * COSINE (MxN)")  # MxM: [all-objects as subject, all-objects as object], MxN: cosine; Output MxN: [all-objects as subject, selected-objects as object]
        pred2tmp1 = {}  # predicate: (MxM) * (MxN) = (MxN)
        for p, mt in tqdm(pred2matrix.items(), total=len(pred2matrix)):
            # print(mt.shape)
            # print(obj_sim.shape)
            pred2tmp1[p] = mt * obj_sim
        if int_files_dir is not None: 
            pkl.dump(pred2tmp1, open(os.path.join(int_files_dir, '1_pred2tmp1_obj.pkl'), 'wb'))
        # multiply the sparse matrix (MxN)' with (MxN)
        if verbose:
            print("Compute  FREQUENCY MATRIX (NxN) = FREQUENCY MATRIX (MxN)' * COSINE (MxN)")  # 
        pred2tmp2 = {}  # predicate: (MxN)' * (MxN) = (NxN)
        for p, mt in tqdm(pred2tmp1.items(), total=len(pred2tmp1)):
            pred2tmp2[p] = torch.matmul(torch.from_numpy(mt).transpose(1, 0).float(), obj_sim).transpose(1, 0)  # newly added transpose to correct to relation direction
        if int_files_dir is not None: 
            pkl.dump(pred2tmp2, open(os.path.join(int_files_dir, '2_pred2tmp2_subj.pkl'), 'wb'))
        # pkl.dump(pred2tmp2, open(os.path.join(logdir, 'pred2tmp2.pkl'), 'wb'))
        # create the matrix L x N x N 
        if verbose:
            print("Prepare LxNxN (L: |All predicates|, N: |Selected objects|)")
        matrix_tmp3 = []
        for p in tqdm(all_preds):
            mt = pred2tmp2.get(p, torch.zeros((self.n_size, self.n_size)))
            matrix_tmp3.append(mt)
        matrix_tmp3 = torch.stack(matrix_tmp3)
        if int_files_dir is not None: 
            pkl.dump(matrix_tmp3, open(os.path.join(int_files_dir, '2_2_matrix_tmp3.pkl'), 'wb'))
        # print(matrix_tmp3.shape)  # torch.Size([22481, 150, 150])
        # multiply with the predicate sim 
        if verbose: 
            print("Compute (PxNxN) = NxNxL * LxP (where P = |selected predicates|)")
        # matrix_tmp4 = torch.from_numpy(np.transpose(matrix_tmp3.numpy(), (1, 2, 0)))  # N x N x L 
        matrix_tmp4 = matrix_tmp3.permute((1, 2, 0))
        if int_files_dir is not None: 
            pkl.dump(matrix_tmp4, open(os.path.join(int_files_dir, '2_3_matrix_tmp4.pkl'), 'wb'))
        # print(matrix_tmp4.shape)
        matrix_tmp5 = torch.matmul(matrix_tmp4, pred_sim)
        if int_files_dir is not None: 
            pkl.dump(matrix_tmp5, open(os.path.join(int_files_dir, '3_matrix_tmp5.pkl'), 'wb'))
        # compute output
        # output = torch.from_numpy(np.transpose(matrix_tmp5.numpy(), (2, 0, 1)))
        output = matrix_tmp5.permute((2, 0, 1))
        output = torch.cat((torch.zeros(1, self.n_size, self.n_size), output), dim=0)  # add zeros for the predicate zero 
        output_paded = F.pad(output, (1, 0, 1, 0), 'constant', 0)  # pad the sim matrix  (first row, first column all zeros)
        if verbose:
            print("Output shape: {}".format(output_paded.shape))
        # pkl.dump(output_paded, open(ofile, 'wb'))
        # print("Saved to", ofile)
        return output_paded

    def prepare_sparse_matrix_for_pred(self, triples, all_objs, pred_list=None):
        # prepare for creating sparse matrix for each predicate 
        m_size = len(all_objs)
        print("Extracting information from all triples")
        pred2info = {}  # predicate: (row, col, data)
        for t, c in tqdm(triples.items(), total=len(triples)):
            s, p, o = t
            if pred_list is not None:
                if s not in all_objs or o not in all_objs or p not in pred_list:  # for running self distillation
                    continue 
            if p not in pred2info:
                pred2info[p] = ([], [], [])
            row, col, data = pred2info[p]
            try: 
                si = all_objs.index(s)
                oi = all_objs.index(o)
            except:
                continue
            row.append(si)
            col.append(oi)
            data.append(c)
        # create sparse matrix for each predicate 
        print("Create sparse matrix for each predicate")
        pred2matrix = {}  # predicate: sparse matrix 
        for p, info in tqdm(pred2info.items(), total=len(pred2info)):
            row, col, data = info
            pred2matrix[p] = csr_matrix((data, (row, col)), shape=(m_size, m_size))
        return pred2matrix

    def get_sparse_matrix_combined(self, triples, all_objs, all_preds):
        '''
        combine all predicate, each predicate is a flattened vector 
        '''
        m_size = len(all_objs)
        print("Extracting information from all triples")
        row = []
        col = []
        data = []
        for t, c in tqdm(triples.items(), total=len(triples)):
            s, p, o = t
            sidx = all_objs.index(s)
            oidx = all_objs.index(o)
            pidx = all_preds.index(p)
            row.append(pidx)
            col.append(sidx * m_size + oidx)
            data.append(c)
        # create sparse matrix for all the data 
        print("Create sparse matrix")
        matrix = csr_matrix((data, (row, col)), shape=(len(all_preds), m_size*m_size))
        return matrix

    def load_sim_files(self, obj_sim_file, pred_sim_file):
        if obj_sim_file is None or not os.path.isfile(obj_sim_file):
            return None, None 
        if pred_sim_file is None or not os.path.isfile(pred_sim_file):
            return None, None 
        print("Load similarity matrices from \n{}\n{}".format(obj_sim_file, pred_sim_file))
        objs_sim = torch.from_numpy(pkl.load(open(obj_sim_file, 'rb')))  # torch.Size([42016, 150])
        preds_sim = torch.from_numpy(pkl.load(open(pred_sim_file, 'rb')))
        return objs_sim, preds_sim

    def filter_triples_by_freq(self, t2c, min_freq):
        tmp = {}
        for k, v in t2c.items():
            if v >= min_freq:
                tmp[k] = v
        return tmp 

    def count_triples(self, train_ids, lemmatize=False):
        t2c = {}  # triple: count 
        lem = None 
        if lemmatize:
            lem = self.lemmatizer
        for img in tqdm(train_ids):
            a = self.img2info[img]
            for r in a['relationships']:
                tp = self.get_triple(r, lemmatizer=lem)
                # if tp[0] == 'short':
                #     print(tp)
                if tp is not None: 
                    t2c[tp] = t2c.get(tp, 0) + 1
        return t2c 

    def get_triple(self, rel, get_raw=False, lemmatizer=None):
        if get_raw:
            return (self.get_name(rel['subject']), rel['predicate'], self.get_name(rel['object']))
        s = self.std_text(self.get_name(rel['subject']), lemmatizer=lemmatizer)
        # p = std_text(dt['predicate'], pos='v')
        p = self.std_text(rel['predicate'], lemmatizer=None)
        o = self.std_text(self.get_name(rel['object']), lemmatizer=lemmatizer)
        if s is None or p is None or o is None:
            return None 
        return (s, p, o)
    
    def get_name(self, obj):
        keys = ['name', 'names']
        for k in keys:
            name = self.get_name_by(obj, k)
            if name is not None:
                if isinstance(name, list):
                    if len(name) > 1:
                        print("More than 1 name: {}".format(name))
                    return name[0]
                return name
        print("Cannot extract `name' for", obj)
        return None 

    def std_text(self, txt, pos='n', lemmatizer=None):
        if txt is None:
            return None
        tmp = txt.strip().lower()
        if lemmatizer is not None:
            return lemmatizer.lemmatize(tmp, pos=pos)
        else:
            return tmp 
        
    def get_name_by(self, obj, key):
        if key not in obj:
            return None
        return obj[key]

class KMeans:
    def __init__(self, model_path, pred_file=None):
        print("Load KMeans model from", model_path)
        self.kmeans = pkl.load(open(model_path, 'rb'))
        self.cluster2imgs = self.load_precomputed_pred(pred_file)
    
    def predict(self, x):
        return self.kmeans.predict(x)
    
    def load_precomputed_pred(self, pred_file):
        if pred_file is  None or not os.path.isfile(pred_file): 
            return None 
        print("Loading pre-computed predictions from", pred_file)
        dt = pkl.load(open(pred_file, 'rb'))
        return self.group_img_by_cluster(dt['img_ids'], dt['prediction'])
    
    def group_img_by_cluster(self, img_ids, predictions):
        clust2imgs = {} 
        for img, c in zip(img_ids, list(predictions)):
            if c not in clust2imgs:
                clust2imgs[c] = []
            clust2imgs[c].append(img)
        return clust2imgs

def mkdirs(idir):
    if not os.path.isdir(idir):
        os.makedirs(idir)

def generate_knowledge_for_all_train_imgs(odir='kg/data/cb_prior_knowledge/all_train_imgs', 
                                          with_distillation=True, int_files_dir=None, self_distillation=False, 
                                          precomputed_pred_sim_file=None, precomputed_obj_sim_file=None):
    
    ke = KnowledgeExtractor(relationship_file=relationship_file, mapping_file=mapping_file, 
                                img_info_file=img_info_file, 
                                embedding_class=embedding_class)
    # for all train images 
    train_ids = json.load(open(split_train))
    # train_ids = [a[0] for a in json.load(open('kg/data/preprocessed/splits/train.json'))]  # old train ids 
    subject2pair_prob, so2pred, freq_matrix = ke.extract_knowledge(train_ids, odir=odir, verbose=False, save_logs=False, lemmatize=True, 
                                   with_distillation=with_distillation, int_files_dir=int_files_dir, 
                                   self_distillation=self_distillation, pred_sim_file=precomputed_pred_sim_file, obj_sim_file=precomputed_obj_sim_file)
    tpf = os.path.join(odir, 'triple_prob.pkl')
    return compute_triple_prob_for_matrix(freq_matrix, ofile=tpf)

def compute_cosine_similarity(X, Y, ofile=None, padding=False):
    print("X shape: {}".format(X.shape))
    print("Y shape: {}".format(Y.shape))
    sims = pw.cosine_similarity(X, Y)  # compute cosine similarity between the objects 
    if padding: 
        sims = F.pad(torch.from_numpy(sims), (1, 0, 1, 0), 'constant', 0)  # pad the sim matrix  (first row, first column all zeros)
    if ofile is not None:
        pkl.dump(sims, open(ofile, 'wb'))
        print("Saved obj_sims to", ofile)
    else:
        return sims

def generate_knowledge_for_clusters(root, 
                                    model_path, 
                                    pred_file, with_distillation=False, 
                                    save_intermediate_files=False, self_distillation=False, 
                                    precomputed_pred_sim_file=None, precomputed_obj_sim_file=None, verbose=False, 
                                    save_logs=False):
    mkdirs(root)
    ke = KnowledgeExtractor(relationship_file=relationship_file, mapping_file=mapping_file, 
                                img_info_file=img_info_file, 
                                embedding_class=embedding_class)
    # for each context (cluster)
    kmeans = KMeans(model_path=model_path, pred_file=pred_file)
    cluster2pairprob = {}  # {cluster: prior_knowledge}
    cluster2so2pred = {} 
    for c, ids in kmeans.cluster2imgs.items():
        print("-----------------------------------------------------------")
        print("Processing cluster {} ({} images)".format(c, len(ids)))
        c = int(c)
        cluster_odir = os.path.join(root, "{}".format(c))
        if save_intermediate_files:
            cluster_inter_dir = os.path.join(cluster_odir, 'intermediate_files')
        else:
            cluster_inter_dir = None 
        prior, so2pred, b = ke.extract_knowledge(ids, odir=cluster_odir, verbose=verbose, 
                                                 save_logs=save_logs, lemmatize=True, with_distillation=with_distillation, 
                                                 int_files_dir=cluster_inter_dir, self_distillation=self_distillation, 
                                                 pred_sim_file=precomputed_pred_sim_file, obj_sim_file=precomputed_obj_sim_file)
        cluster2pairprob[c] = prior    
        cluster2so2pred[c] = so2pred
        tmp = {'cluster2pair_prob': cluster2pairprob, 'cluster2pred_prob': cluster2so2pred}
        json.dump(tmp, open(os.path.join(root, 'cluster2prior_probs.json'), 'w'))
        # pkl.dump(tmp, open(os.path.join(root, 'cluster2knowledge.pkl'), 'wb'))
        # pkl.dump(cluster2knowledge, open(os.path.join(root, 'cluster2so2pred.pkl'), 'wb'))
    return compute_triple_prob_cluster(idir=root, filename=freq_matrix_filename)

def compute_triple_prob_for_matrix(matrix, ofile=None, excludes_rels=None):
    dtso = torch.sum(matrix, dim=0)
    pair_prob = normalize(dtso, p=1, dim=1)  #   normalize by object P(o|s)  including "exclude_rels" (frequent predicates)
    if excludes_rels is not None: 
        matrix2 = matrix.clone()
        matrix2[excludes_rels] = 0
        pred_prob = normalize(matrix2, p=1, dim=0)  # normalize predicate without frequent predicates 
    else:
        pred_prob = normalize(matrix, p=1, dim=0)  # normalize predicate 
    a = pair_prob.expand_as(pred_prob)
    triple_prob = a*pred_prob
    if ofile is not None: 
        pkl.dump(triple_prob, open(ofile, 'wb'))
        print("Saved to", ofile)
        return ofile
    return triple_prob

def compute_triple_prob_single_file(ifile, ofile=None, excludes_rels=None):
    dt = pkl.load(open(ifile, 'rb'))
    return compute_triple_prob_for_matrix(dt, ofile, excludes_rels=excludes_rels)

def compute_triple_prob_cluster(idir, num_cluster=25, filename='freq_matrix.pkl', 
                                o_filename='cluster_triple_prob.pkl', excludes_rels=None):
    ofile = os.path.join(idir, o_filename)
    c2tp = {}  # cluster to triple prob 
    # i = 22
    for i in range(num_cluster):
        ifile = os.path.join(idir, str(i), filename)
        triple_prob = compute_triple_prob_single_file(ifile, excludes_rels=excludes_rels)
        c2tp[i] = triple_prob
    pkl.dump(c2tp, open(ofile, 'wb'))
    print("Saved to", ofile)
    return ofile
    # if return_result:
    #     return c2tp


def prepare_top_pred_for_subj_obj(dt, idx_to_label, idx_to_pred, rel_prob_threshold = 0.1, rel_freq_threshold = 3):
    pred_norm = normalize(dt, p=1, dim=0)
    so2pred = {}  # {subj: {obj: pred}}
    for s in range(1, 151):
        stxt = idx_to_label[str(s)]
        for o in range(1, 151):
            otxt = idx_to_label[str(o)]
            a = pred_norm[:, s, o]
            p = torch.argmax(a).item()  # top predicate index 
            # check constraint 
            if p == 0:
                # print("(p=0) No relation: ({}, {}, p: {})".format(idx_to_label[str(s)], idx_to_label[str(o)], p))
                continue
            if a[p] < rel_prob_threshold or dt[p, s, o] < rel_freq_threshold:
                # print("No relation: ({}, {}, {}, prob: {}, freq: {})".format(idx_to_label[str(s)], idx_to_pred[str(p)], idx_to_label[str(o)], a[p], dt[p,s,o]))
                continue
            else:  # satisfy
                if stxt not in so2pred:
                    so2pred[stxt] = {}
                so2pred[stxt][otxt] = idx_to_pred[str(p)]
    return so2pred

def run_knowledge_extraction(root_dir, 
                             kmeans_model_path=None, kmeans_prediction_path=None,
                             with_distillation=True, save_intermediate_files=False, 
                             run_cluster=True, self_distillation=False, 
                             precomputed_pred_sim_file=None, precomputed_obj_sim_file=None):
    # generate for clusters 
    if run_cluster:
        # for now the precomputed predicate sim is the same for different context 
        # NOTE: currently don't save intermediate files for cluster 
        return generate_knowledge_for_clusters(root=os.path.join(root_dir, 'context_based'), 
                                        model_path=kmeans_model_path, 
                                        pred_file=kmeans_prediction_path, with_distillation=with_distillation, 
                                        save_intermediate_files=save_intermediate_files, self_distillation=self_distillation, 
                                        precomputed_pred_sim_file=precomputed_pred_sim_file, precomputed_obj_sim_file=precomputed_obj_sim_file)
    else: # 28/8/2025: copied from run_no_context 
        int_files_dir = None 
        all_image_dir = os.path.join(root_dir, 'no_context')
        if save_intermediate_files:
            int_files_dir = os.path.join(all_image_dir, 'intermediate_files')
        return generate_knowledge_for_all_train_imgs(odir=all_image_dir, with_distillation=with_distillation, 
                                            int_files_dir=int_files_dir, self_distillation=self_distillation, 
                                            precomputed_obj_sim_file=precomputed_obj_sim_file, precomputed_pred_sim_file=precomputed_pred_sim_file)
        
def get_args():
    parser = argparse.ArgumentParser(description="Argument parser for SRD")
    parser.add_argument("--output_dir", type=str, default="srd_data/output",
                        help="Output dir")
    parser.add_argument("--relationship_file", type=str, default="srd_data/relationships.json",
                        help="Path to relationship file")
    parser.add_argument("--mapping_file", type=str, default="srd_data/VG-SGG-dicts.json",
                        help="Path to mapping file")
    parser.add_argument("--img_info_file", type=str, default="srd_data/all_triplets.json",
                        help="Path to image info file")
    parser.add_argument("--embedding_class", type=str, default="sentence_transformer",
                        help="Embedding class to use")
    parser.add_argument("--kmeans_model_path", type=str, default="srd_data/output/kmeans_25.pkl",
                        help="Path to KMeans model file")
    parser.add_argument("--kmeans_train_prediction_path", type=str, 
                        default="srd_data/output/kmeans_25_prediction.pkl",
                        help="Path to KMeans train prediction file")
    parser.add_argument("--split_train", type=str, default="srd_data/train.json",
                        help="Path to train split file")
    return parser.parse_args()


def print_header(header):
    print("=" * 40)
    print(header)
    print("=" * 40)

if __name__ == "__main__":
    args = get_args()
    print(args)
    
    output_dir = args.output_dir
    relationship_file = args.relationship_file
    mapping_file = args.mapping_file
    img_info_file=args.img_info_file
    embedding_class=args.embedding_class
    kmeans_model_path = args.kmeans_model_path
    kmeans_train_prediction_path = args.kmeans_train_prediction_path
    split_train = args.split_train
    freq_matrix_filename = 'freq_matrix.pkl'
    print_header("Compute No-Context Triple Probability")
    no_context_triple = run_knowledge_extraction(root_dir=output_dir, 
                                                    with_distillation=True, save_intermediate_files=False, run_cluster=False, self_distillation=False)
    
    print_header("Compute Context-based Triple Probability")
    context_based_triple = run_knowledge_extraction(root_dir=output_dir, 
                                                    kmeans_model_path=kmeans_model_path, 
                                                    kmeans_prediction_path=kmeans_train_prediction_path, 
                                                    with_distillation=True, save_intermediate_files=False, run_cluster=True, self_distillation=False)
   
    print_header(f"[OUTPUT] No-Context Triple Probability: {no_context_triple}; Context-based Triple Probability: {context_based_triple}")
