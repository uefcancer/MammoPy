import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from skimage import feature
import torch
import matplotlib.pyplot as plt
import pydicom
from PIL import Image
import mammopy as mi


class analysis():
        
    MASK_COLORS = ["red", "green", "blue", "yellow", "magenta", "cyan"]
    def mask_to_rgba(mask, color="red"):
        """
        Converts binary segmentation mask from white to red color.
        Also adds alpha channel to make black background transparent.
        Args:
            mask (numpy.ndarray): [description]
            color (str, optional): Check `MASK_COLORS` for available colors. Defaults to "red".
        Returns:
            numpy.ndarray: [description]
        """
        assert color in analysis.MASK_COLORS
        assert mask.ndim == 3 or mask.ndim == 2

        h = mask.shape[0]
        w = mask.shape[1]
        zeros = np.zeros((h, w))
        ones = mask.reshape(h, w)
        if color == "red":
            return np.stack((ones, zeros, zeros, ones), axis=-1)
        elif color == "green":
            return np.stack((zeros, ones, zeros, ones), axis=-1)
        elif color == "blue":
            return np.stack((zeros, zeros, ones, ones), axis=-1)
        elif color == "yellow":
            return np.stack((ones, ones, zeros, ones), axis=-1)
        elif color == "magenta":
            return np.stack((ones, zeros, ones, ones), axis=-1)
        elif color == "cyan":
            return np.stack((zeros, ones, ones, ones), axis=-1)

    def density_estimation(model, img):
        """
        Computes the percentage of mammogram density in an input image.
        Args:
            model (str): The name of the pre-trained model to use.
            img: Input image in np.ndarray or PIL.Image format
            
        Returns:
            - image: The image in 256*256 size used for analysis
            - pred1: The breast area of mammogram
            - pred2: The dense area of mammogram
            - percentage_density: The percentage of mammogram density in the input image.
        Raises:
            TypeError: If `model` is not a string or `image_path` is not a string.
            ValueError: If `model` is not a valid pre-trained model name or `image_path`
            does not point to a valid image file.
        """
        
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Model must be a PyTorch model.")
        if type(img) not in [np.ndarray, Image.Image]:
            raise TypeError("Input must be np.ndarray or PIL.Image")
        
        
        img = mi.pre_process.image_tensor(img)
        # Compute the mammogram density
        
        pred1, pred2 = model.module.predict(img)

        img = img[0].cpu().numpy().transpose(1, 2, 0)
        img = img[:, :, 0]

        pred1 = pred1[0].cpu().numpy().transpose(1, 2, 0)
        pred1 = pred1[:, :, 0]

        pred2 = pred2[0].cpu().numpy().transpose(1, 2, 0)
        pred2 = pred2[:, :, 0]

        breast_area = np.sum(np.array(pred1) == 1)
        dense_area = np.sum(np.array(pred2) == 1)
        density = (dense_area / breast_area) * 100
        density = np.rint(density)

        return img, pred1, pred2, density


    def single_density_estimation(model, img, visualization = False, breast_segmentation=False, dense_segmentation=False):
        """
        Computes the percentage of mammogram density in an input image.
        Args:
            model (str): The name of the pre-trained model to use.
            img: Input image in np.ndarray or PIL.Image format
            visualization: For display output
            breast_segmentation: Customize output to visualize Breast Area
            dense_segmentation: Customize output to visualize Dense Area
        Returns:
            dict: A dictionary containing the results of the computation. The dictionary
            has the following keys:
                - non_dense_area: The total number of pixels in the input image that corresponse 
                to non-dense tissue. (i.e., non-fibroglandular)
                - dense_area: The total number of pixels in the input image that corresponse
                to dense tissue. (i.e., fibroglandular)
                - percentage_density: The percentage of mammogram density in the input image.
        Raises:
            TypeError: If `model` is not a string or `image_path` is not a string.
            ValueError: If `model` is not a valid pre-trained model name or `image_path`
            does not point to a valid image file.
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Model must be a PyTorch model.")
        if type(img) not in [np.ndarray, Image.Image]:
            raise TypeError("Input must be np.ndarray or PIL.Image")
        elif (visualization != True) and ((breast_segmentation == True) or (dense_segmentation == True)):
            raise ValueError("For display option Visualization must be true")
        
        # Compute the mammogram density
        result = {}

        img, pred1, pred2, _ = mi.analysis.density_estimation(model, img)

        breast_area = np.sum(np.array(pred1) == 1)
        dense_area = np.sum(np.array(pred2) == 1)
        density = (dense_area / breast_area) * 100
        density = np.rint(density)
        
        # Populate the result dictionary with the computed values
        result["Non_Dense_Area"] = breast_area
        result["Dense_Area"] = dense_area
        result["Percentage_Density"] = density

        if visualization == False:
            return result
        elif (visualization == True) and (breast_segmentation == True) and (dense_segmentation == False):
            edges = analysis.canny_edges(pred1)
            plt.title('Breast area contour')
            plt.imshow(img, cmap='gray')
            plt.imshow(analysis.mask_to_rgba(edges, color='red'), cmap='gray')
            plt.axis('off')
            plt.show()
            return result
        elif (visualization == True) and (breast_segmentation == False) and (dense_segmentation == True):
            plt.title('Dense tissues')
            plt.imshow(img, cmap='gray')
            plt.imshow(analysis.mask_to_rgba(pred2, color='green'), cmap='gray')
            plt.axis('off')
            plt.show()
            return result
        else:
            edges = analysis.canny_edges(pred1)
            fig, axes = plt.subplots(1,2, figsize = (15,10),squeeze=False)
            axes[0, 0].set_title('Image', fontsize=16)
            axes[0, 1].set_title("Breast and dense tissue segmentation", fontsize=20)
            axes[0, 0].imshow(img, cmap='gray')
            axes[0, 0].set_axis_off()
            axes[0, 1].imshow(img, cmap='gray')
            axes[0, 1].imshow(analysis.mask_to_rgba(edges, color='red'), cmap='gray')
            axes[0, 1].imshow(analysis.mask_to_rgba(pred2, color='green'), cmap='gray', alpha=0.7)
            axes[0, 1].set_axis_off()
            plt.show()
            return result

    def canny_edges(image_array):
        """
        Detect edges using the Canny algorithm.
        
        Parameters:
        image_array (numpy.ndarray): The input image array
        
        Returns:
        edges (numpy.ndarray): A binary edge map with detected edges marked with 1.
        
        Raises:
        TypeError: If the input image is not a numpy array.
        """
        if not isinstance(image_array, np.ndarray):
            raise TypeError("Input image must be a numpy array.")
        # Detect edges using Canny algorithm with sigma value of 3.
        edges = feature.canny(image_array, sigma=3)
        return edges

    def multi_density_estimation(model, folder_path):

        """
        Computes the percentage of mammogram density in an input image.
        Args:
            model (str): The name of the pre-trained model to use.
            folder_path (str): The path to the input folder of files.
        Returns:
            dict: A dictionary containing the results of the computation of each valid file. The dictionary
            has the following keys:
                - File: Name of the valid file, Skips the analysis for wrong file format
                - Non_Dense_Area: The total number of pixels in the input file that corresponse 
                to non-dense tissue. (i.e., non-fibroglandular)
                - Dense_Area: The total number of pixels in the input file that corresponse
                to dense tissue. (i.e., fibroglandular)
                - Percentage_Density: The percentage of mammogram density of the input files.
        Raises:
            TypeError: If `model` is not a string or `folder_path` is not a string.
            ValueError: If `model` is not a valid pre-trained model name or `folder_path`
            does not point to a valid folder.
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Model must be a PyTorch model.")
    
        if os.path.exists(folder_path):
            files = os.scandir(folder_path)
        else:
            raise FileExistsError("Folder path must be a string with '/' forward slashs")
        result = []
        for file in files:
            if file.is_file():
                file_ext = file.name.split(".")[-1]
                if file_ext in ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG', 'tif', 'tiff', 'TIF', 'TIFF']:
                    print(file.name)
                    img = Image.open(open(file, 'rb')).convert('RGB')
                    temp = analysis.single_density_estimation(model=model, img=img)
                elif file_ext == 'dcm' or file_ext == 'DCM':
                    dicom_file = pydicom.dcmread(file)
                    pil_image = Image.fromarray(dicom_file.pixel_array)
                    img = pil_image.convert('RGB')
                    temp = analysis.single_density_estimation(model=model, img=img)
                else:
                    try:
                        ds = pydicom.dcmread(file)
                        pixel_array  = ds.pixel_array
                        if ds.PhotometricInterpretation == 'MONOCHROME1':
                            pixel_array = (pixel_array * 255.0 / np.max(pixel_array)).astype(np.uint8)
                            pil_image = Image.fromarray(pixel_array)
                            img = pil_image.convert('RGB')
                            img = np.max(img) - img
                            np_to_PILimg = Image.fromarray(img)
                            temp = analysis.single_density_estimation(model=model, img=np_to_PILimg)
                        else:
                            temp = None
                            pass
                    except pydicom.errors.InvalidDicomError:
                        temp = None
                        pass
        # Populate the result dictionary with the computed values
            if temp is None:
                pass
            else:
                temp = {'File': file.name, "Non_Dense_Area": temp["Non_Dense_Area"], 'Dense_Area': temp["Dense_Area"], 'Percentage_Density': temp["Percentage_Density"]}
                result.append(temp)
        return result

