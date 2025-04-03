import pickle
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from transformers import AutoProcessor, LlavaForConditionalGeneration

import sys, os
import base64
import openai
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

sys.path.append(os.getcwd())
from maskrcnn_benchmark.utils.logger import setup_logger

BOX_SCALE = 1024

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
ORG_ID = os.environ['ORG_ID']


def remove_whitespace(input_str):
    input_str = input_str.replace("    ", "")
    input_str = input_str.replace("\t", "")
    return input_str


def read_image(image_input):
    if type(image_input) == str:
        image = Image.open(image_input).convert("RGB")
    elif type(image_input) == np.ndarray:
        image = Image.fromarray(image_input)
    else: # Image.Image
        image = image_input
    return image


class LLAVA_hf:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf"):
        # https://huggingface.co/docs/transformers/main/en/model_doc/llava
        # https://huggingface.co/docs/transformers/main/en/model_doc/llava_next
        self.name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name)

        self.logger = setup_logger("llava", ".", 0, filename="llava_log.txt")


    def response(self, prompt: str, image: Image.Image):
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        generate_ids = self.model.generate(**inputs, max_new_tokens=500)
        output = self.processor.batch_decode(generate_ids, skip_special_tokens=True, 
                                                clean_up_tokenization_spaces=False)[0]
        return output


    def gen_new_relation_prompt_batch(self, pair_idx, rel_dict):
        prompt = f"""USER: <image>\n \
        Choose one predicate to be the most possible relation for every subject-object pair {pair_idx} \
        from the predicate dictionary {rel_dict}. \
        Note, if the subject-object pair is unlikely to have any relations, choose "__background__" as their relation. \
        Ouput only the predicate/relation as a Python list.
        ASSISTANT:
        """
        prompt = remove_whitespace(prompt)
        return prompt
    

    def gen_new_relation_prompt(self, pair, rel_dict):
        prompt = f"""USER: <image>\n \
        Choose one predicate to be the most possible relation for the subject-object pair {pair} \
        from the predicate dictionary {rel_dict}. \
        Note, if the subject-object pair is unlikely to have any relations, choose "__background__" as their relation. \
        Ouput only the predicate/relation as a Python list.
        ASSISTANT:
        """

        prompt = remove_whitespace(prompt)
        return prompt


    def gen_new_relation_prompt_with_box(self, pair, pair_boxes, rel_dict):
        sbj_dict = {"subject": pair[0], "bounding_box": pair_boxes[0]}
        obj_dict = {"object": pair[1], "bounding_box": pair_boxes[1]}
        prompt = f"""USER: <image>\n \
        Choose one predicate to be the most possible relation for the subject-object pair {sbj_dict}, \
        {obj_dict} from the predicate dictionary {rel_dict}. \
        Note, if the subject-object pair is unlikely to have any relations, choose "__background__" as their relation. \
        Ouput only the predicate/relation as a Python list.
        ASSISTANT:
        """
        prompt = remove_whitespace(prompt)
        return prompt


class ChatGPT:
    def __init__(self, model_name="gpt-4o"):
        self.name = model_name
        self.client = OpenAI(organization=ORG_ID)
        # record usage
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.logger = setup_logger(model_name, ".", 0, filename="chatgpt_log.txt")


    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    def gen_new_relation_prompt_with_box(self, pair, pair_boxes, rel_dict):
        sbj_dict = {"subject": pair[0], "bounding_box": pair_boxes[0]}
        obj_dict = {"object": pair[1], "bounding_box": pair_boxes[1]}
        prompt = f"""
        Choose one predicate to be the most possible relation for the subject-object pair {sbj_dict}, \
        {obj_dict} from the predicate dictionary {rel_dict}. \
        Note, if the subject-object pair is unlikely to have any relations, choose "__background__" as their relation. \
        Ouput only the predicate/relation as a Python list.
        """
        prompt = remove_whitespace(prompt)
        return prompt


    def response(self, prompt, base64_image):
        response = self.client.chat.completions.create(
            model=self.name,
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "auto"
                    },
                    },
                ],
                }
            ],
            max_tokens=500,
        )
        # print(response.choices[0])
        self.update_usage(response)
        return response.choices[0].message.content


    def response_chatgpt35(self, prompt):
        response = self.client.chat.completions.create(
            model=self.name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
        )
        # print(response.choices[0].message)
        self.update_usage(response)
        return response.choices[0].message.content
    

    def update_usage(self, response):
        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        return



def draw_single_box(img, box, color='red', draw_info=None):
    draw = ImageDraw.Draw(img)
    x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)
    if draw_info:
        draw.rectangle(((x1, y1), (x1+50, y1+10)), fill=color)
        info = draw_info
        draw.text((x1, y1), info)
    img.save("show_box.png")



def load_vg_data(path="vg_train_transformer_N50_f1o1.pkl", 
                 logits_path="output/relabel-transformer/sup_data_with_transformer_prediction.pk"):
    """filenames, gt_boxes, gt_objects, relationships, 
    ind_to_classes, ind_to_predicates, new_triple"""
    data = pickle.load(open(path, "rb"))
    data_and_logits = pickle.load(open(logits_path, "rb"))
    return data, data_and_logits


def getitem(data, data_and_logits, index, use_bg=False):
    filename = data["filenames"][index]
    image = read_image(filename)
    w, h = image.width, image.height

    ind_to_classes = data["ind_to_classes"]

    # object list of strings
    objects = [ind_to_classes[i] for i in data["gt_objects"][index]]
    # convert bbox to strings
    boxes = data["gt_boxes"][index].copy() / BOX_SCALE * max(w, h) # actual box size
    # draw_single_box(image, boxes[0], draw_info=objects[0])
    bbox_str = []
    for i in range(boxes.shape[0]):
        box = boxes[i]
        bbox_str.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])]) # xyxy
    
    # candidate pairs has been filtered by our SRD deduction
    new_rel = data["new_triple"][index]
    if new_rel is not None and len(new_rel) != 0:
        pair_idx = new_rel[:, :2]
    else:
        if use_bg:
            # use randomly sampled bg pairs
            image_id = int(filename.split(".")[0].split("/")[-1])
            curr_dict = data_and_logits[image_id].copy()
            pair_idx = curr_dict["rel_pair_idxs"] # [num_pairs, 2]
            num_gt_rel = curr_dict["triples"].shape[0]
            all_pair_idx = pair_idx[num_gt_rel:]
            idxes = random.sample(range(len(all_pair_idx)), 5)
            pair_idx = [all_pair_idx[idx] for idx in idxes]
        else:
            pair_idx = []

    pairs = [[objects[pair[0]], objects[pair[1]]] 
             for pair in pair_idx]
    pair_boxes = [[bbox_str[pair[0]], bbox_str[pair[1]]]
                  for pair in pair_idx]
    return filename, image, bbox_str, pair_idx, pairs, pair_boxes


def get_unique_objects(data, index):
    ind_to_classes = data["ind_to_classes"]
    # object list of strings
    objects = []
    for i in data["gt_objects"][index]:
        if ind_to_classes[i] not in ["person", "man", "men", "woman", "child", "boy", "people", "lady", "guy"]:
            objects.append(ind_to_classes[i])
    return list(set(objects))


def count_num_images(data):
    total = len(data["new_triple"]) # 57723
    num_image_to_relabel = 0
    num_pair_to_relabel = 0
    for i in range(total):
        new_rel = data["new_triple"][i]
        if new_rel is not None and len(new_rel) != 0:
            num_image_to_relabel += 1
            num_pair_to_relabel += new_rel.shape[0]
    print(num_image_to_relabel, num_pair_to_relabel) # 21790 41763
    return num_image_to_relabel, num_pair_to_relabel


def create_relation_manifest(data, save_name="transformer_relation_all.csv"):
    df = pd.DataFrame()
    rel_dict = vg["ind_to_predicates"]
    for i in range(len(data["filenames"])):
        filename, image, bbox_str, pair_idx, pairs, pair_boxes = getitem(vg, None, i)
        if len(pairs) == 0:
            continue
        
        new_rel = data["new_triple"][i][:, -1]
        for j in range(len(pairs)):
            row = {'filename': filename, 
                    'subject': pairs[j][0],
                    'object': pairs[j][1],
                    'relation': rel_dict[new_rel[j]],
                    'subject_box': ','.join(map(str, pair_boxes[j][0])),
                    'object_box': ','.join(map(str, pair_boxes[j][1]))}
            df = df._append(row, ignore_index=True)
    df.to_csv(save_name)
    print("Saved relation manifest at", save_name)


def random_sample_relations(path, num_sample):
    all_data = pd.read_csv(path)
    samples = all_data.sample(num_sample)
    samples.to_csv(f"sample_{num_sample}.csv")
    print(f"Sampled {num_sample} to a csv file")


def create_object_manifest(data, save_name="vg_objects.csv"):
    df = pd.DataFrame()
    image_cnt = 0

    for i in range(len(data["filenames"])):
        filename = data["filenames"][i]
        objs = get_unique_objects(vg, i)
        if len(objs) == 0:
            continue
        image_cnt += 1
        row = {'filename': filename, 'objects': objs}
        df = df._append(row, ignore_index=True)
        
        if image_cnt >= 100:
            break
    df.to_csv(save_name)
    print("Saved object manifest at", save_name)



if __name__=='__main__':
    logits_path = "output/relabel-transformer/sup_data_with_transformer_prediction.pk"
    vg, data_and_logits = load_vg_data(path="vg_train_transformer_N25_f15o1.pkl",
                                       logits_path=logits_path)
    # save all relation and infos to a csv file.
    # create_relation_manifest(vg)
    # random_sample_relations("demo/transformer_relation_all.csv", 300)
    # create_object_manifest(vg)

    rel_dict = vg["ind_to_predicates"]
    count_num_images(vg)

    llm = ChatGPT("gpt-4o") # [LLAVA_hf, ChatGPT("gpt-3.5-turbo")]
    batch_prompt = False

    count = 0
    for i in range(100):
        filename, image, bbox_str, pair_idx, pairs, pair_boxes = getitem(vg, data_and_logits, i)
        if len(pairs) == 0:
            continue

        llm.logger.info(filename)
        if batch_prompt:
            prompt = llm.gen_new_relation_prompt_batch(pairs, rel_dict)
            response = llm.response(prompt, image)
            llm.logger.info(response)
        else:
            for j in range(len(pairs)):
                # prompt = llm.gen_new_relation_prompt(pairs[j], rel_dict)
                prompt = llm.gen_new_relation_prompt_with_box(pairs[j], pair_boxes[j], rel_dict)
                if type(llm) == ChatGPT:
                    llm.logger.info(prompt)
                    if llm.name == "gpt-4o":
                        base64_image = llm.encode_image(filename)
                        response = llm.response(prompt, base64_image)
                    else: # gpt3.5
                        response = llm.response_chatgpt35(prompt)
                else: # llava
                    response = llm.response(prompt, image)
                llm.logger.info(response)

        count += 1
        if count >= 10:
            if type(llm) == ChatGPT:
                llm.logger.info(f"prompt_tokens: {llm.prompt_tokens}, \
                                completion_tokens: {llm.completion_tokens}")
            break
    # print(max_w, max_h) # 1280 1280
