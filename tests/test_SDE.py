import unittest
import sys, os
sys.path.insert(0, os.path.abspath('.'))
import mammopy as mi
from PIL import Image
import numpy as np


class ImageTensor(unittest.TestCase):

################################################################
################ Tensor function testing #######################
################################################################
    def test_image_tesor(self):
        #####################
        ### Input 1 for test
        # arrange
        #mode = mi.pre_process.load_model('base')
        self.model = mi.pre_process.load_model('base')

        self.image = Image.open('random_images/ran_test_img_06.jpg').convert('RGB')
        self.img = mi.pre_process.image_tensor(self.image)
        pred1, pred2 = self.model.module.predict(self.img)
        pred1 = pred1[0].cpu().numpy().transpose(1, 2, 0)
        pred1 = pred1[:, :, 0]
        pred2 = pred2[0].cpu().numpy().transpose(1, 2, 0)
        pred2 = pred2[:, :, 0]

        breast_area = np.sum(np.array(pred1) == 1)
        dense_area = np.sum(np.array(pred2) == 1)
        density = (dense_area / breast_area) * 100
        density = np.rint(density)
        result = {}
        # Populate the result dictionary with the computed values
        result["Non_Dense_Area"] = breast_area
        result["Dense_Area"] = dense_area
        result["Percentage_Density"] = density
        print(result)
        temp_res = mi.analysis.single_density_estimation(self.model, self.image)
        print(temp_res)
        
        #Assert
        self.assertEqual(result["Non_Dense_Area"], temp_res["Non_Dense_Area"])

    def test_types(self):
        self.image2 = 'random_images/ran_test_img_02.jpg'
        self.assertRaises(TypeError, mi.pre_process.image_tensor, 'abcd')
        self.assertRaises(TypeError, mi.pre_process.image_tensor, 2584)
        self.assertRaises(TypeError, mi.pre_process.image_tensor, 0.258)
        self.assertRaises(TypeError, mi.pre_process.image_tensor, self.image2)



if __name__ =='__main__':
    unittest.main()