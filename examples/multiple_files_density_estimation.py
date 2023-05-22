import sys, os
sys.path.insert(0, os.path.abspath('.'))
import mammopy as mi

# loading model
model = mi.pre_process.load_model('base')

# load Mammogram folder
# two methods to give address of folder
# No. 1 with forward slash '/'
#mamo_folder = 'random_images/'
# No. 2 complete address with back slash '\'
mamo_folder = r'random_images'

# density estimation of multiple images
temp_MDE = mi.analysis.multi_density_estimation(model=model, folder_path=mamo_folder)

# output :
#   [{'File': 'filename.jpg', 'Non_Dense_Area': 0000, 'Dense_Area': 000, 'Percentage_Density': 0.0}, {'File': 'filename.jpg', 'Non_Dense_Area': 0000, 'Dense_Area': 000, 'Percentage_Density': 0.0}, ...]
print(temp_MDE)