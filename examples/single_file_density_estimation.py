import sys, os
sys.path.insert(0, os.path.abspath('.'))
import mammopy as mi
from PIL import Image

# loading model
model = mi.pre_process.load_model('base')

# load Mammogram
img = 'random_images/ran_test_img_04.jpg'

# convert in PIL.Image format
image = Image.open(img).convert('RGB')

# density estimation (visuals are optional)
# Visuals:
#   visualization= True:
#       for visualization it must be true.
#   breast_segmentation= True:
#       if only breast segmentation are required  to display.
#   dense_segmentation=True:
#       if only density segmentation are required to display.
temp_SDE = mi.analysis.single_density_estimation(model, image)


print(temp_SDE)