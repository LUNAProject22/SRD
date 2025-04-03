import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def draw_single_box(img, box, color='red', draw_info=None):
    if img.mode == "L": # grayscale image
        img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=5)
    if draw_info:
        font = ImageFont.load_default(size=15)
        textbox_w = len(draw_info) * 8
        draw.rectangle(((x1, y1), (x1+textbox_w, y1+20)), fill=color)
        draw.text((x1, y1), draw_info, font=font)
    # img.save("show_box.png")
    return img


def resize_image_with_aspect_ratio(image, new_width):
    """
    Resize a PIL image while maintaining its aspect ratio.

    :param image: PIL Image object
    :param new_width: New width in pixels
    :return: Resized PIL Image object
    """
    width, height = image.size
    new_height = int(height * (new_width / width))
    
    # Resize the image using the new dimensions
    resized_image = image.resize((new_width, new_height))
    return resized_image


def process_image(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    new_df = pd.DataFrame()
    for index, row in df.iterrows():
        filename = row["filename"]
        image_name = filename.split("/")[-1]
        # create a new name for visualization image
        image_id, ext = os.path.splitext(image_name)
        new_name = f"{image_id}_{index}{ext}"
        relation = row["relation"]
        subject_box = row["subject_box"].split(",")
        object_box = row["object_box"].split(",")
        img = Image.open(filename)
        draw_single_box(img, subject_box, color=(255,51,255,100), draw_info=row["subject"]) # pink
        draw_single_box(img, object_box, color=(51,153,255,100), draw_info=row["object"]) # blue
        # resize image 
        img = resize_image_with_aspect_ratio(img, new_width=500)
        img.save(os.path.join(save_dir, new_name))
        img.close()
        row["poll_image"] = os.path.join(save_dir, new_name)
        new_df = new_df._append(row, ignore_index=True)
    # update csv file
    new_df.to_csv("demo/sample_new.csv", index=False)
    return



if __name__=='__main__':
    samples = pd.read_csv("demo/sample_300.csv")
    process_image(samples, save_dir="demo/poll_image")
