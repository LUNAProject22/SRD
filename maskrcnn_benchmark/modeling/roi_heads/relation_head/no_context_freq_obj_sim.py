'''
20/09/2022:
using the no-context frequency, but accumulated based on object-object similarity

'''
import pickle as pkl
import json


class RelPredictor_NoContext:
    def __init__(self, prior_file='data/kg_inference/prior_knowledge.pkl',
                 index_file='data/VisualGnome/preprocessed_codebase/VG-SGG-dicts.json', num_classes=151,
                 knowledge_key='matrix_rht_sim_paraphrase-mpnet-base-v2'):
        self.priors = pkl.load(open(prior_file, 'rb'))  #
        self.matrix = self.priors[knowledge_key]
        self.max_idx = num_classes - 1
        self.label_to_idx, self.idx_to_label, self.predicate_to_idx, self.idx_to_predicate = load_mappings(index_file)


    def verify_idx(self, idx):
        if idx < 0 or idx > self.max_idx:
            idx = 0
        return idx

    def predict(self, head, tail):
        if isinstance(head, str) and isinstance(tail, str):
            # head and tail is index
            head = head.strip()
            tail = tail.strip()
            head = self.label_to_idx.get(head, 0)
            tail = self.label_to_idx.get(tail, 0)
        # head = self.verify_idx(head)
        # tail = self.verify_idx(tail)
        vec = self.matrix[:, head, tail]
        return vec.permute(1, 0)

def load_mappings(index_file):
    data = json.load(open(index_file))
    label_to_idx = data['label_to_idx']
    idx_to_label = data['idx_to_label']
    predicate_to_idx = data['predicate_to_idx']
    idx_to_predicate = data['idx_to_predicate']
    return convert_to_dict(label_to_idx, value_as_int=True), convert_to_dict(idx_to_label, key_as_int=True), \
           convert_to_dict(predicate_to_idx, value_as_int=True), convert_to_dict(idx_to_predicate, key_as_int=True)


def convert_to_dict(idict, key_as_int=False, value_as_int=False):
    tmp = {}
    for k, v in idict.items():
        if key_as_int:
            k = int(k)
        if value_as_int:
            v = int(v)
        tmp[k] = v
    return tmp


if __name__ == "__main__":
    predictor = RelPredictor_NoContext(prior_file='datasets/kg/prior_knowledge_0920.pkl',
                                       index_file='datasets/vg/VG-SGG-dicts.json',
                                       # knowledge_key='p(t,h,r)',
                                       knowledge_key='matrix_rht_sim_paraphrase-mpnet-base-v2'  # 20/09/2022
                                       )
    vec = predictor.predict(head='logo', tail='shirt')
    print(vec)

    # dt = pickle.load(open('data/kg_inference/prior_knowledge.pkl', 'rb'))
    # dt.keys()
    # a = dt['p(t,h,r)']
    # a[1][1]
    # a.shape