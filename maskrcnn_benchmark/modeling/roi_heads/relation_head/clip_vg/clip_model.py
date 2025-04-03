import os
import torch
import json
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer
from .file_utils import pt_load
from .region_prompt_generator import get_cpt_one
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip
from torchvision.transforms.functional import InterpolationMode, hflip
import numpy as np

def compute_iou(box1, box2):
    # box = [x1, y1, x2, y2]
    # return iou
    x1 = box1[0]
    y1 = box1[1]
    x2 = box1[2]
    y2 = box1[3]

    x3 = box2[0]
    y3 = box2[1]
    x4 = box2[2]
    y4 = box2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (x2 - x1) * (y2 - y1)
    bb2_area = (x4 - x3) * (y4 - y3)

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


class transform_test(object):
    def __init__(self, n_px, data_mean, data_std) -> None:
        self.resize = Resize((n_px, n_px), interpolation=InterpolationMode.BICUBIC) # Enforce Region or Image to Square
        self.image2rgb = image2rgb()
        self.to_tensor = ToTensor()
        self.norm = Normalize(data_mean, data_std)


    def __call__(self, image):
        image = self.resize(image)
        image = self.image2rgb(image)
        image = self.to_tensor(image)
        image = self.norm(image)
        return image


class image2rgb(object):
    # Convert image to 
    def __call__(self, image):
        return image.convert("RGB")


class openclip_relation():
    def __init__(self, model_name, model_path, vg_dict_json=None, image_folder=None):
        # 1. Load Model
        model, _, _ = create_model_and_transforms(
            model_name,
            #pretrained='openai',
            precision='fp32',
            output_dict=True,
            )

        self.toknizer = get_tokenizer(model_name)
        model = model.cuda()
        model = model.eval()
        checkpoint = pt_load(model_path, map_location='cpu')
        sd = checkpoint["state_dict"]
        start_epoch = checkpoint["epoch"]
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        if next(iter(sd.items()))[0].startswith('_orig_mod'):
            sd = {k[len('_orig_mod.module.'):]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        print(f"=> resuming checkpoint '{model_path}' (epoch {start_epoch})")
        self.model = model

        # 2. Load VGG Dict and Image Folder
        self.image_folder = image_folder
        self.vgg_dict_json = json.load(open(vg_dict_json, 'r'))
        self.idx_to_class = self.vgg_dict_json['idx_to_label']
        self.idx_to_predicate = self.vgg_dict_json['idx_to_predicate']
        self.preprocess = transform_test(224, [0.48145466, 0.4578275, 0.40821073],
         [0.26862954, 0.26130258, 0.27577711])


    def gen_all_relations(self, subject_category, object_category):
        count = len(self.idx_to_predicate)
        all_relations = []
        all_relations.append(subject_category + " no relation " + object_category)
        for ii in range(count):
            str_ii = "{}".format(ii+1)
            all_relations.append(subject_category + " " + self.idx_to_predicate[str_ii] + " " + object_category)
        return all_relations


    def color_prompt(self, o_image, subject_box, object_box):
        highlight_boxes = [
            {
                'left': subject_box[0],
                'top': subject_box[1],
                'width': subject_box[2] - subject_box[0],
                'height': subject_box[3] - subject_box[1],
            },
            {
                'left': object_box[0],
                'top': object_box[1],
                'width': object_box[2] - object_box[0],
                'height': object_box[3] - object_box[1],
            }
        ]

        image = get_cpt_one(o_image, highlight_boxes)
        return image


    def image_to_torch_tensor_v2(self, image, bboxes_image):
        # randomly select one box during training;
        # use specified one box during testing
        image = self.preprocess(image)
        return image, bboxes_image


    def predict_one_image(self, image_name, box_list, object_list, subject_object_list=-1*torch.ones((1, 2))):
        # Input format
        # category_list of length: 
        #           [category1, category2, ..., categoryN]
        # box_list of length N: 
        #           [[x1, y1, x2, y2], 
        #            [x1, y1, x2, y2],
        #            ...
        #            [x1, y1, x2, y2]]
        # subject_object_list of length M: 
        #           [[subject_index_1, object_index_1],
        #            [subject_index_2, object_index_2],
        #                 ...
        #            [subject_index_M, object_index_M]]
        #
        # Return format:
        #  Detected_List:
        #           [subject_index_1, object_index_1, relation_index_1, confidence_1],
        #           [subject_index_2, object_index_2, relation_index_2, confidence_2],
        #           ...
        #           [subject_index_M, object_index_M, relation_index_M, confidence_M]

        box_list = box_list.cpu().numpy()
        object_list = object_list.cpu().numpy()
        subject_object_list = subject_object_list.cpu().numpy()

        Detected_List = []
        Detected_List_Probs = []
    
        # Read Image
        image_path = os.path.join(self.image_folder, 'VG_100K', image_name)
        if not os.path.exists(image_path):
            image_path = image_path.replace('VG_100K', 'VG_100K_2')
        o_image = Image.open(image_path)


        if subject_object_list.sum() < 0:
            # Create subject_object_list by pair-wising box_list (N x N - N-> remove diagonal elements)
            subject_object_list = []
            N = len(box_list)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        # calculate IoU
                        subject_box = box_list[i]
                        object_box = box_list[j]
                        iou = compute_iou(subject_box, object_box)
                        if iou > 0.01:
                            subject_object_list.append([i, j])
        if len(subject_object_list) == 0:
            Detected_List.append( [0, 0 ])
            probs = torch.zeros(51).cuda()
            probs[0] = 1
            Detected_List_Probs.append(probs)

            return Detected_List, Detected_List_Probs
        # Detect relations for one pair of subject and object
        for kk, subject_object in enumerate(subject_object_list):
            subject_index = subject_object[0]
            object_index  = subject_object[1]

            subject_box   = box_list[subject_index] # box = [x1, y1, x2, y2]
            object_box    = box_list[object_index]  # box = [x1, y1, x2, y2]
            subject_category_idx = "{}".format(object_list[subject_index]) # category id
            object_category_idx  = "{}".format(object_list[object_index])  # category id

            w, h = o_image.size
            box_scale_factor =  max(w, h) / 1024.0
            subject_box = subject_box * box_scale_factor
            object_box = object_box * box_scale_factor

            # 1. Given a pair, generate all relations
            subject_category = self.idx_to_class[subject_category_idx]
            object_category  = self.idx_to_class[object_category_idx]

            all_relations = self.gen_all_relations(subject_category, object_category)
            all_relations = self.toknizer(all_relations).cuda()


            # 2. Apply colourful prompt to subject and object boxes.
            image = self.color_prompt(o_image, subject_box, object_box)
            # save image with kk naming
            #print("DEBUG: {} {}".format(subject_category, object_category))
            #image = image.convert("RGB")
            #image.save("color_prompt_{}_{}_{}.jpg".format(subject_category, self.idx_to_predicate[str(subject_object[2])], object_category))

            image,_ = self.image_to_torch_tensor_v2(image, None)
            # convert to bf16
            image = image.unsqueeze(0).cuda()
            # image = image.bfloat16().cuda()
            all_relations = all_relations.long().cuda()

            image_features = self.model.encode_image(image, normalize=True)
            all_relations_features = self.model.encode_text(all_relations, normalize=True)
            
            text_probs = (100.0 * image_features @ all_relations_features.T).softmax(dim=-1)
            # find max index and max value
            max_index = torch.argmax(text_probs)
            max_value = torch.max(text_probs)
            #print("DEBUG: {} {} {} {}".format(subject_index, object_index, max_index, max_value 
            #))
            #print("DEBUG: {} all_relations = {}".format(subject_object, all_relations))
            #Detected_List.append([subject_index, object_index, max_index.item(), max_value.item()])
            Detected_List.append([subject_index, object_index])
            # Detected_List_Probs.append(text_probs.cpu().float())
            Detected_List_Probs.append(text_probs.float())


        # Sort Detected_List by confidence
        if len(Detected_List) == 0:
            Detected_List.append( [0, 0 ])
            probs = torch.zeros(51).cuda()
            probs[0] = 1
            Detected_List_Probs.append(probs)
        #Detected_List = np.array(Detected_List)
        #Detected_List = Detected_List[Detected_List[:, 3].argsort()[::-1]]
        ## To List
        #Detected_List = Detected_List.tolist()
        ## first 3 elements to int
        #for i in range(len(Detected_List)):
        #    Detected_List[i][0] = int(Detected_List[i][0])
        #    Detected_List[i][1] = int(Detected_List[i][1])
        #    Detected_List[i][2] = int(Detected_List[i][2])
        return Detected_List, Detected_List_Probs
                            
