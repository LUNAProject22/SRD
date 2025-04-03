from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
from textwrap import wrap
import random
from torchvision.transforms import Resize

# Single (ID=2): CPT only (CPT)
def get_cpt_one(o_image, bboxes):
	# Draw Colorful Prompt on image
	image= o_image.convert('RGBA') #highlight mode
	overlay = Image.new('RGBA', image.size, '#00000000')
	draw = ImageDraw.Draw(overlay, 'RGBA')

	for ii, bbox in enumerate(bboxes):
		x = bbox['left']
		y = bbox['top']
		if ii % 2 == 0:
			draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
                              fill='#ff05cd3c', outline='#05ff37ff', width=3)
		else:
			draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
							  fill='#0000ff3c', outline='#05ff37ff', width=3)

	image = Image.alpha_composite(image, overlay)
	overlay.close()

	return image


