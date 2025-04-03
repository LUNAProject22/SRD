import torch
from PIL import Image
import clip

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.data.datasets.visual_genome import BOX_SCALE, VGDataset


def CLIP_visual_encoding(model, preprocess, image):
    model.cuda().eval()
    image = preprocess(image)
    image_input = torch.tensor(image).cuda().unsqueeze(dim=0)
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.squeeze()
    return image_features


def get_CLIP_embedding(split, img_dir, roidb_file, dict_file, image_file, logits_cfg=None):
    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()

    data = VGDataset(split=split, img_dir=img_dir, roidb_file=roidb_file,
                    dict_file=dict_file, image_file=image_file, num_val_im=5000, logits_cfg=logits_cfg)
    filenames = data.filenames
    image_info = data.img_info

    clip_image_features = {}
    for filename, info in zip(filenames, image_info):
        image_id = info['image_id']
        image = Image.open(filename).convert("RGB")
        image_features = CLIP_visual_encoding(model, preprocess, image)
        clip_image_features[image_id] = image_features
    torch.save(clip_image_features, 'output/clip_image_feature_{}.pth'.format(data.split))


def gt_to_boxlist(gt_boxes, gt_classes, img_info, filename):
    w, h = img_info['width'], img_info['height']
    # important: recover original box from BOX_SCALE
    box = gt_boxes / BOX_SCALE * max(w, h)
    box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes

    target = BoxList(box, (w, h), 'xyxy') # xyxy
    target.add_field("labels", torch.from_numpy(gt_classes))
    target = target.clip_to_image(remove_empty=True)
    # target.add_field("image_path", filename)
    return target, box


def get_CLIP_region_embedding(img_dir, roidb_file, dict_file, image_file, logits_cfg=None):
    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()

    train_data = VGDataset(split='train', img_dir=img_dir, roidb_file=roidb_file,
                        dict_file=dict_file, image_file=image_file, num_val_im=5000, logits_cfg=logits_cfg)
    filenames = train_data.filenames
    gt_classes = train_data.gt_classes
    gt_boxes = train_data.gt_boxes
    img_infos = train_data.img_info

    boxlists = {}
    for filename, gt_class, gt_boxes, img_info in zip(filenames, gt_classes, gt_boxes, img_infos):
        boxlist, box = gt_to_boxlist(gt_boxes, gt_class, img_info, filename)
        # crop obj region and encode feature
        image = Image.open(filename).convert("RGB")
        features = torch.zeros((box.shape[0], 512))
        for i in range(box.shape[0]):
            b = (int(box[i, 0]), int(box[i, 1]), int(box[i, 2]), int(box[i, 3]))
            crop = image.crop(b)
            # print(ind_to_classes[gt_classes[i]])
            # crop.show()
            features[i] = CLIP_visual_encoding(model, preprocess, crop)
        boxlist.add_field("clip_feature", features)

        boxlists[filename] = boxlist
    torch.save(boxlists, 'output/boxlists_with_clip_obj_feature.pth')



if __name__=='__main__':
    from yacs.config import CfgNode as CN
    logits_cfg = CN()
    logits_cfg.ENABLE = True
    logits_cfg.SPECIFIED_DATA_FILE="datasets/vg/vg_sup_data.pk"
    get_CLIP_embedding(split="train", img_dir="datasets/vg/VG_100K", roidb_file="datasets/vg/VG-SGG-with-attri.h5", \
                dict_file="datasets/vg/VG-SGG-dicts-with-attri.json", image_file="datasets/vg/image_data.json", \
                logits_cfg=logits_cfg)

    # get_CLIP_region_embedding(img_dir="datasets/vg/VG_100K", roidb_file="datasets/vg/VG-SGG-with-attri.h5", \
    #             dict_file="datasets/vg/VG-SGG-dicts-with-attri.json", image_file="datasets/vg/image_data.json")