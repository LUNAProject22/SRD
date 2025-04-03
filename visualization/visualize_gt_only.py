import pickle
import json
import argparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.lines as lines
from tqdm import tqdm
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument(
        '--output_dir', default='./output/',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')
    parser.add_argument(
        '--num',
        help='Visualization Number',
        default=5, type=int)
    parser.add_argument(
        '--no_do_vis',
        help='do not visualize',
        action='store_true')
    parser.add_argument(
        '--rel_class_recall', help='rel class recall.',
        action='store_true')
    parser.add_argument(
        '--phrdet', help='use phrdet.',
        action='store_true')
    parser.add_argument(
        '--dataset',
        help='Visualization Number',
        default='ag', type=str)    
    parser.add_argument(
        '--filename',
        help='Visualization file',
        default='data/cache/ag_val_rel_gt_roidb.pkl', type=str) 
    parser.add_argument(
        '--cnt_lim',
        help='Visualization Number',
        default=10, type=int)
    parser.add_argument(
        '--lim',
        help='Visualization Number',
        default=0, type=int)
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.output_dir, 'vis_gt')):
        os.mkdir(os.path.join(args.output_dir, 'vis_gt'))
    saved_path = os.path.join(args.output_dir, 'vis_gt')

    with open(args.filename, 'rb') as f:
        roidb = pickle.load(f)
        f.close()
    
    with open(os.path.join('data', args.dataset,'annotations/objects.json'), 'r') as f:
        obj_list = json.load(f)
        f.close()
    with open(os.path.join('data', args.dataset, 'annotations/predicates.json'), 'r') as f:
        rel_list = json.load(f)
        f.close()
    
    edge_width = 3
    font_size = 18

    print('total {} images. '.format(len(roidb)))
    args.num = min(args.num, len(roidb))
    print('Number is {}. '.format(args.num))
    cnt = 0
    for roidb_i in roidb[::20]:
        f_name = roidb_i['image']
        gt_boxes_sbj = roidb_i['sbj_gt_boxes']
        gt_boxes_obj = roidb_i['obj_gt_boxes']
        gt_labels_sbj = roidb_i['sbj_gt_classes']
        gt_labels_obj = roidb_i['obj_gt_classes']
        gt_labels_prd = roidb_i['prd_gt_classes']
        gt_boxes_so = np.hstack((gt_boxes_sbj, gt_boxes_obj))
        gt_labels_spo = np.vstack((gt_labels_sbj, gt_labels_prd, gt_labels_obj)).transpose()
        

        if len(gt_labels_prd) > args.lim and cnt <= args.cnt_lim:
            if not args.no_do_vis:
                saved_name = f_name.split('/')[-2:]
                saved_name = saved_name[0] + '/' + saved_name[1]
                img = mpimg.imread(f_name)

                rec_pos = {}
                fig = plt.figure(figsize=(18, 12))
                ax = plt.gca()
                plt.imshow(img)
                plt.axis('off')
                det_title = plt.title('Annotation')
                plt.setp(det_title, color='b')
                for i in range(len(gt_boxes_sbj)):
                    x, y, x1, y1 = gt_boxes_sbj[i].astype(np.int)
                    
                    s_name = obj_list[gt_labels_sbj[i]]
                    
                    s_cx, s_cy = (x+x1)//2, (y+y1)//2 
                    
                    srect = plt.Rectangle((x,y),x1-x,y1-y, fill=False, edgecolor='b', linewidth=3)
                    ax.add_patch(srect)
                    
                    ax.text(x, y,
                        s_name,
                        fontsize=font_size,
                        color='white',
                        bbox=dict(facecolor='orange', alpha=0.5, pad=0, edgecolor='none'))
                    
                    
                    x, y, x1, y1 = gt_boxes_obj[i].astype(np.int)
                    o_name = obj_list[gt_labels_obj[i]]
                    o_cx, o_cy = (x+x1)//2, (y+y1)//2 
                    
                    orect = plt.Rectangle((x,y),x1-x,y1-y, fill=False, edgecolor='b', linewidth=3)
                    ax.add_patch(orect)
                    
                    ax.text(x, y,
                        o_name,
                        fontsize=font_size,
                        color='white',
                        bbox=dict(facecolor='blue', alpha=0.5, pad=0, edgecolor='none'))
                    
                    p_name = rel_list[gt_labels_prd[i].astype(np.int)]
                    
                    rel_l = lines.Line2D([s_cx, o_cx], [s_cy, o_cy], color='purple', linewidth=3)
                    ax.add_line(rel_l)
                    
                    lx, ly = s_cx + 8*(o_cx - s_cx) / 9, s_cy + 8*(o_cy - s_cy) / 9
                    
                    if (lx, ly) in rec_pos:
                        rec_pos[(lx, ly)] += 10
                    else:
                        rec_pos[(lx, ly)] = 0
                    d = rec_pos[(lx, ly)]
                        
                    ax.text(lx, ly + d,
                            p_name,
                            fontsize=font_size,
                            color='white',
                            bbox=dict(facecolor='purple', alpha=0.5, pad=0, edgecolor='none'))
                    
                saved_file_name = saved_name.replace('/', '_')
                plt.savefig(os.path.join(saved_path, saved_file_name), bbox_inches='tight')
                plt.close(fig)
