import sys, os
sys.path.insert(0, os.path.abspath('.'))
import mammopy as mi
from PIL import Image

# load Mammogram
img = 'random_images/ran_test_img_04.jpg'

# convert in PIL.Image format
image = Image.open(img).convert('RGB')

# convert in tensor shape
temp_tensor = mi.pre_process.image_tensor(image)

# tensor shape should be in [1, 3, 256, 256]
print(temp_tensor.shape)