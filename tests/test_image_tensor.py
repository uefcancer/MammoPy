# Path hack.
import sys, os
sys.path.insert(0, os.path.abspath('.'))
import unittest
from torchvision import transforms
import mammopy as mi
from PIL import Image


class ImageTensor(unittest.TestCase):

################################################################
################ Tensor function testing #######################
################################################################
    def test_image_tesor(self):
        #####################
        ### Input 1 for test
        # arrange
        self.image = Image.open('random_images/ran_test_img_04.jpg').convert('RGB')
        torch_tensor = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
        self.image_tens = torch_tensor(self.image)
        self.image_tens = self.image_tens.unsqueeze(0)
        # action
        img = mi.pre_process.image_tensor(self.image)
        #Assert
        self.assertEqual(self.image_tens.shape, img.shape)
        

    def test_types(self):
        self.image2 = 'random_images/EE0B1627'
        self.assertRaises(TypeError, mi.pre_process.image_tensor, 'abcd')
        self.assertRaises(TypeError, mi.pre_process.image_tensor, 2584)
        self.assertRaises(TypeError, mi.pre_process.image_tensor, 0.258)
        self.assertRaises(TypeError, mi.pre_process.image_tensor, self.image2)



if __name__ =='__main__':
    unittest.main()