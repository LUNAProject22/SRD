import os
import numpy as np
from tqdm import tqdm

from maskrcnn_benchmark.utils.miscellaneous import filter_new_triples, count_triples


triple_paths = {
    "transformer": [
        "output/relabel-transformer/N5/relation_cache_transformer_pretrain_idf_N5_CD.npy",
        "output/relabel-transformer/N10/relation_cache_transformer_pretrain_idf_N10_CD.npy",
        "output/relabel-transformer/N15/relation_cache_transformer_pretrain_idf_N15_CD.npy",
        "output/relabel-transformer/N20/relation_cache_transformer_pretrain_tfidf_N20_CD_train.npy",
        "output/relabel-transformer/N25/relation_cache_transformer_pretrain_tfidf_N25_CD_train.npy",
        "output/relabel-transformer/N30/relation_cache_transformer_pretrain_tfidf_N30_CD_train.npy",
        "output/relabel-transformer/N35/relation_cache_transformer_pretrain_tfidf_N35_CD_train.npy",
        "output/relabel-transformer/N40/relation_cache_transformer_pretrain_tfidf_N40_CD_train.npy",
        "output/relabel-transformer/N45/relation_cache_transformer_pretrain_tfidf_N45_CD_train.npy",
        "output/relabel-transformer/N50/relation_cache_transformer_pretrain_tfidf_N50_CD_train.npy"
    ],
    "motif": [
        "output/relabel-motif/N5/relation_cache_motif_pretrain_idf_N5_CD.npy",
        "output/relabel-motif/N10/relation_cache_motif_pretrain_idf_N10_CD.npy",
        "output/relabel-motif/N15/relation_cache_motif_pretrain_idf_N15_CD.npy",
        "output/relabel-motif/N20/relation_cache_motif_pretrain_idf_N20_CD.npy",
        "output/relabel-motif/N25/relation_cache_motif_pretrain_tfidf_N25_CD_train.npy",
        "output/relabel-motif/N30/relation_cache_motif_pretrain_idf_N30_CD.npy",
        "output/relabel-motif/N35/relation_cache_motif_pretrain_idf_N35_CD.npy",
        "output/relabel-motif/N40/relation_cache_motif_pretrain_idf_N40_CD.npy",
        "output/relabel-motif/N45/relation_cache_motif_pretrain_idf_N45_CD.npy",
        "output/relabel-motif/N50/relation_cache_motif_pretrain_idf_N50_CD.npy"
    ],
    "vctree": [
        "output/relabel-vctree/N5/relation_cache_vctree_pretrain_idf_N5_CD.npy",
        "output/relabel-vctree/N10/relation_cache_vctree_pretrain_idf_N10_CD.npy",
        "output/relabel-vctree/N15/relation_cache_vctree_pretrain_idf_N15_CD.npy",
        "output/relabel-vctree/N20/relation_cache_vctree_pretrain_idf_N20_CD.npy",
        "output/relabel-vctree/N25/relation_cache_vctree_pretrain_tfidf_N25_CD_train.npy",
        "output/relabel-vctree/N30/relation_cache_vctree_pretrain_idf_N30_CD.npy",
        "output/relabel-vctree/N35/relation_cache_vctree_pretrain_idf_N35_CD.npy",
        "output/relabel-vctree/N40/relation_cache_vctree_pretrain_idf_N40_CD.npy",
        "output/relabel-vctree/N45/relation_cache_vctree_pretrain_idf_N45_CD.npy",
        "output/relabel-vctree/N50/relation_cache_vctree_pretrain_idf_N50_CD.npy"
    ]
}


def merge_triples(paths, outfile):
    merged_data = np.load(paths[0], allow_pickle=True)[()]
    init_triple_cnt = count_triples(merged_data)
    print(f"first triple: {init_triple_cnt}")
    for path in paths[1:]:
        curr_data = np.load(path, allow_pickle=True)[()]
        # iter through image level
        for i in tqdm(range(len(merged_data))):
            old_triple = merged_data[i]
            new_triple = curr_data[i]
            if new_triple is None or new_triple.shape[0] == 0:
                continue
            if old_triple is None:
                merged_data[i] = np.array(new_triple, dtype=np.int32)
                continue
            old_triple, filtered_new_triple = filter_new_triples(old_triple, new_triple)
            if filtered_new_triple is not None and filtered_new_triple.shape[0] > 0:
                relation = np.vstack((old_triple, filtered_new_triple)) # (num_rel, 3)
                merged_data[i] = relation
        curr_triple_cnt = count_triples(merged_data)
        print(f"added triple: {curr_triple_cnt - init_triple_cnt}")
        init_triple_cnt = curr_triple_cnt
    print(f"Final triple: {count_triples(merged_data)}")
    np.save(outfile, merged_data, allow_pickle=True)
    print("Saved relation cache file:", outfile)
    return merged_data


if __name__=='__main__':
    model_name = "transformer"
    data_dir = f"output/relabel-{model_name}/merge_n"
    os.makedirs(data_dir, exist_ok=True)
    outfile = f"merged_{model_name}_N5-25_F1"
    outfile = os.path.join(data_dir, outfile)
    ds = merge_triples(triple_paths[model_name][:5], outfile=outfile)
