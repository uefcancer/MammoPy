import numpy as np
import requests
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class pre_process():
    def load_model(model_path):
        """
        Loads a pre-trained model from a given path.
        Args:
        - model_path: str, the path to the pre-trained model. 
                    If model_path = 'base', then the default pre-trained model will be loaded.
        
        Returns:
        - model: torch.nn.Module, the pre-trained model loaded from the specified path.
        Raises:
        - TypeError: if the input model_path is not 'base'.
        """
        
        if model_path == 'base':
            # Define the path to the default pre-trained model weights
            model_weights_path = "weights.pth"
            path = Path(model_weights_path)

            # If the model weights file does not exist, download it from Dropbox
            if not path.is_file():
                url = "https://www.dropbox.com/s/37rtedwwdslz9w6/all_datasets.pth?dl=1"
                response = requests.get(url)
                open("weights.pth", "wb").write(response.content)
            
            # Load the pre-trained model and set the device to CPU
            model = torch.load(model_weights_path, map_location=torch.device("cpu"))
            # Convert the model to be compatible with multiple GPUs, if available
            model = nn.DataParallel(model.module)
            
            return model
        else:
            raise TypeError("Model not implemented")

    def image_tensor(img):
        """
        Converts a PIL Image or NumPy array to a PyTorch tensor.
        Args:
        - img: PIL.Image or np.ndarray, the image to be converted to a PyTorch tensor.
        Returns:
        - image: torch.Tensor, the converted PyTorch tensor.
        Raises:
        - TypeError: if the input image is not a PIL.Image or np.ndarray.
        """
        
        if type(img) not in [np.ndarray, Image.Image]:
            raise TypeError("Input must be np.ndarray or PIL.Image")

        # Define a PyTorch tensor transformer pipeline
        torch_tensor = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor()]
        )

        if type(img) == Image.Image:
            # Convert PIL image to PyTorch tensor
            image = torch_tensor(img)
            image = image.unsqueeze(0)
            #print("tensor", type(image))
            return image
        elif type(img) == np.ndarray:
            # Convert NumPy array to RGB PIL image and then to PyTorch tensor
            pil_image = Image.fromarray(img).convert("RGB")
            image = torch_tensor(pil_image)
            image = image.unsqueeze(0)
            #print("tensor", type(image))
            return image
        else:
            raise TypeError("Input must be np.ndarray or PIL.Image")

