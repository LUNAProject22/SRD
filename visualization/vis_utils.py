import re
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


def print_list(name, input_list):
    for i, item in enumerate(input_list):
        print(name + ' ' + str(i) + ': ' + str(item))


def draw_single_box(pic, box, color=(100,255,100), draw_info=None): # green
    fnt = ImageFont.truetype("demo/STXINWEI.TTF", size=22)
    draw = ImageDraw.Draw(pic)
    x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=2)
    if draw_info:
        # draw.rectangle(((x1, y1), (x1+14*len(draw_info), y1+20)), fill=color)
        draw.text((x1+3, y1), draw_info, fill=color, font=fnt)


def draw_object(img_path, boxes, labels):
    pic = Image.open(img_path)
    num_obj = boxes.shape[0]
    for i in range(num_obj):
        info = str(i) + "-" + labels[i]
        draw_single_box(pic, boxes[i], draw_info=info)
    return pic


def draw_relation(pic, boxes, relation_dict, triples, labels, color):
    """Draw relation on images"""
    fnt_large = ImageFont.truetype("demo/STXINWEI.TTF", size=20)
    num_rel = triples.shape[0]
    for i in range(num_rel):
        sbj_upleft_pt = boxes[triples[i][0]][:2].numpy()
        obj_upleft_pt = boxes[triples[i][1]][:2].numpy()
        text_pt = (sbj_upleft_pt + obj_upleft_pt) / 2
        rel = triples[i][2]
        info = relation_dict[rel]
        draw = ImageDraw.Draw(pic)
        draw.line([tuple(sbj_upleft_pt), tuple(obj_upleft_pt)], fill=color, width=3)
        draw.rectangle(((text_pt[0], text_pt[1]), (text_pt[0]+min(14*len(info),100), text_pt[1]+20)), fill=color)
        draw.text((text_pt[0]+3, text_pt[1]), text=info, fill=(0,0,0), font=fnt_large)
        # draw_single_box(pic, boxes[i], draw_info=str(triples[i][0])+"-"+labels[triples[i][0]])
        # draw_single_box(pic, boxes[i], draw_info=str(triples[i][1])+"-"+labels[triples[i][1]])
        # if labels is not None:
        #     print('{}, {}, {}'.format(str(triples[i][0])+"-"+labels[triples[i][0]], info, str(triples[i][1])+"-"+labels[triples[i][1]]))
    return pic


def rel_idx_to_name(triples, labels, relation_dict):
    num_rel = triples.shape[0]
    res = []
    for i in range(num_rel):
        pred = relation_dict[triples[i][2]]
        rel_name = (labels[triples[i][0]], pred, labels[triples[i][1]])
        res.append(rel_name)
    return res


def load_predicates(dict_file="datasets/vg/VG-SGG-dicts-with-attri.json"):
    info = json.load(open(dict_file, 'r'))
    ind_to_predicates = info['idx_to_predicate']
    return ind_to_predicates

def plot_predicate_recall(input, R=50, save_name="", legend=["ours"]):
    """Input: list of '(above:0.1837) (across:0.2817) (against:0.2984) (along:0.3456) ...'"""
    ind_to_predicates = load_predicates()
    freq_rels = [31, 20, 22, 30, 48, 29, 50, 1, 21, 8, 43, 40, 49, 41, 23,
                38, 6, 7, 33, 11, 46, 16, 25, 47, 19, 5, 9, 35, 24, 10, 4, 14, 
                13, 12, 36, 44, 42, 32, 2, 28, 26, 45, 3, 17, 18, 34, 27, 37, 39, 15]
    freq_rel_text = [ind_to_predicates[str(i)] for i in freq_rels]
    
    fig = plt.subplots(figsize =(12, 8))
    barWidth = 0.8
    br_p = np.arange(len(freq_rel_text))*len(input)
    color_set = ["gray", "brown", "cyan", "olive", "pink"]
    assert type(input) == list
    for i, s in enumerate(input):
        l = re.findall(r'\(.*?\)', s)
        assert len(l) == 50
        recall_dict = {}
        for ss in l:
            if ss[0] == "(" and ss[-1] == ")":
                ss = ss[1:-1]
            pred, recall = ss.split(":")
            recall_dict[pred] = float(recall)
        recall_to_plot = []
        for label in freq_rel_text:
            recall_to_plot.append(recall_dict[label])
            print(recall_dict[label])
        br = [x + barWidth*i for x in br_p]
        plt.bar(br, recall_to_plot, label=legend[i], color=color_set[i], width=barWidth)
    plt.xlabel('Predicates', fontsize=15) 
    plt.ylabel(f'PredCls Recall@{R}', fontsize=15)
    plt.legend()
    plt.savefig(save_name)


if __name__=="__main__":
    R = 50
    model = "transformer"
    recall_data = "(above:0.1837) (across:0.2817) (against:0.2984) (along:0.3456) (and:0.3649) (at:0.6541) (attached to:0.1653) (behind:0.4395) (belonging to:0.4538) (between:0.2583) (carrying:0.6969) (covered in:0.5669) (covering:0.5040) (eating:0.7294) (flying in:0.0000) (for:0.3870) (from:0.4404) (growing on:0.3371) (hanging from:0.4266) (has:0.5059) (holding:0.3993) (in:0.2926) (in front of:0.2773) (laying on:0.4600) (looking at:0.1965) (lying on:0.2321) (made of:0.4062) (mounted on:0.2153) (near:0.0764) (of:0.3637) (on:0.1214) (on back of:0.5225) (over:0.2448) (painted on:0.4483) (parked on:0.8919) (part of:0.1358) (playing:0.1591) (riding:0.8201) (says:0.4167) (sitting on:0.4663) (standing on:0.2885) (to:0.4549) (under:0.3435) (using:0.5642) (walking in:0.4155) (walking on:0.6422) (watching:0.5280) (wearing:0.5319) (wears:0.4117) (with:0.2429) "
    input = [recall_data]
    legend = ["placeholder"]
    save_name = f"output/recall/{model}_R{R}.jpg"
    plot_predicate_recall(input, R=R, legend=legend, save_name=save_name)

