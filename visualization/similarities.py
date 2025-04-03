# for rebuttal 
import pickle as pkl
import json
import numpy as np
import matplotlib.pylab as plt


def obj150_idx(idx_to_label, all_objs):
    indexs = []
    for i in range(1, len(idx_to_label.keys())+1):
        name = idx_to_label[str(i)]
        idx = all_objs.index(name)
        indexs.append(idx)
    return indexs


def extract_obj_sim_150(objs_sim, obj150_id):
    sim = objs_sim.transpose() # [150, 42016]
    obj150_id = np.array(obj150_id)
    res = sim[:, obj150_id]
    assert res.shape[0] == 150 and res.shape[1] == 150
    return res


def top2_most_freq_rel_given_subject(target, target_obj, freq):
    # find freq
    target_freq = freq[:, target, :].clone().squeeze() # 51, 151
    max_xy = np.where(target_freq == target_freq.max())
    max_idx = [max_xy[0][0], max_xy[1][0]]
    print("Most frequent triple:",
          target_obj, idx_to_predicate[str(max_idx[0])], idx_to_label[str(max_idx[1])],
          int(target_freq[max_idx[0], max_idx[1]])) # banana on table
    # second most freq triple
    target_freq[max_idx[0], max_idx[1]] = 0
    second_freq_xy = np.where(target_freq == target_freq.max())
    max_idx = [second_freq_xy[0][0], second_freq_xy[1][0]]
    print("Second most frequent triple:",
          target_obj, idx_to_predicate[str(max_idx[0])], idx_to_label[str(max_idx[1])],
          int(target_freq[max_idx[0], max_idx[1]])) # banana in bowl
    return


def heatmap2d(target, target_obj, freq):
    target_freq = freq[:, target, :].clone().squeeze() # 51, 151
    target_freq = target_freq[1:, 1:]
    plt.imshow(target_freq, cmap='viridis')
    plt.colorbar()
    plt.show()
    plt.savefig(f'{target_obj}.png')
    plt.close()


def heatmap2d_share_axis(target, target_obj, target2, target_obj2, freq):
    crop_y1, crop_y2 = 5, 35
    crop_x1, crop_x2 = 10, 60
    target_freq = freq[:, target, :].clone().squeeze() # 51, 151
    target_freq = target_freq[1+crop_y1:1+crop_y2, 1+crop_x1:1+crop_x2]
    target_freq2 = freq[:, target2,2 :].clone().squeeze() # 51, 151
    target_freq2 = target_freq2[1+crop_y1:1+crop_y2, 1+crop_x1:1+crop_x2]
    # Create a figure object
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(6, 3))

    # Plot the first heatmap
    cax1 = ax1.imshow(target_freq, aspect='auto', cmap='viridis',
                      extent=[crop_x1, crop_x2, crop_y2, crop_y1])
    fig.colorbar(cax1, ax=ax1, orientation='vertical')
    ax1.set_title(target_obj)
    ax1.set_xlabel('Object categories')
    ax1.set_ylabel('Predicate categories')

    # Plot the second heatmap
    cax2 = ax2.imshow(target_freq2, aspect='auto', cmap='viridis',
                      extent=[crop_x1, crop_x2, crop_y2, crop_y1])
    fig.colorbar(cax2, ax=ax2, orientation='vertical')
    ax2.set_title(target_obj2)

    # Set the x-axis labels for the shared axis
    ax2.set_xlabel('Object categories')
    # ax2.set_ylabel('Predicate categories')

    # Adjust layout for better visualization
    plt.tight_layout()
    plt.show()
    plt.savefig(f'frequency_heatmap_{target_obj}.png')
    plt.close()


mappings = json.load(open('datasets/vg/VG-SGG-dicts.json'))
idx_to_label = mappings['idx_to_label']
idx_to_predicate = mappings['idx_to_predicate']
label_to_idx = mappings['label_to_idx']
predicate_to_idx = mappings['predicate_to_idx']

ifile = '/home/username/workspace/SRD/kg/data/cb_prior_knowledge/no_aggregating_freq/all_train_imgs/freq_matrix.pkl'
freq = pkl.load(open(ifile, 'rb')) # [51, 151, 151]

input_lists = json.load(open('/home/username/workspace/SRD/kg/data/preprocessed/similarities/objs_preds_lists_for_sim.json'))
all_objs = input_lists['all_objs'] # 42016
all_preds = input_lists['all_preds'] # 23524

objs_sim = pkl.load(open('/home/username/workspace/SRD/kg/data/preprocessed/similarities/all-mpnet-base-v2_objs_sim.pkl', 'rb'))
# (42016, 150)
preds_sim = pkl.load(open('/home/username/workspace/SRD/kg/data/preprocessed/similarities/all-mpnet-base-v2_preds_sim.pkl', 'rb'))
# (23524, 50)

# locate 150 obj in all obj
obj150_id = obj150_idx(idx_to_label, all_objs)
obj150_similarity = extract_obj_sim_150(objs_sim, obj150_id) # (150, 150)

# car-26, truck-137, vehicle-142
# banana-5, fruit-51, vegetable-141, orange-86

target_obj = "tree"
target_label = label_to_idx[target_obj] # 5


# target_label_in_all_obj = all_objs.index(target_obj)
# assert all_objs[target_label_in_all_obj] == target_obj
# target_sim_vec = objs_sim.transpose()[target_label-1] # 42016
# # zero out self obj
# target_sim_vec[target_label_in_all_obj] = 0

# # find similar obj from all classes
# top_k = np.argsort(target_sim_vec)[::-1][:10]
# top_k_sim_obj = [all_objs[k] for k in top_k]
# print(top_k_sim_obj)

# find similar obj within 150 classes
target_sim_vec_150 = obj150_similarity[target_label-1]
target_sim_vec_150[target_label-1] = 0

top_k = np.argsort(target_sim_vec_150)[::-1][:10]
top_k_sim_obj = [idx_to_label[str(k+1)] for k in top_k]
print(top_k_sim_obj)

most_similar_label = top_k[0] + 1
most_similar_obj = top_k_sim_obj[0]

top2_most_freq_rel_given_subject(target_label, target_obj, freq)
heatmap2d(target_label, target_obj, freq)
## check similar object relations
top2_most_freq_rel_given_subject(most_similar_label, most_similar_obj, freq)

heatmap2d_share_axis(target_label, target_obj, most_similar_label, most_similar_obj, freq)

