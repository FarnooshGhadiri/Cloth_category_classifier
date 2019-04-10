import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

"""
def get_transformer(img_size, crop_size, mean, std):
    transform_list = []
    transform_list.append(transforms.Resize( (crop_size, crop_size), Image.BICUBIC))
    #transform_list.append(transforms.RandomCrop(crop_size))
    #transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize( (mean, mean, mean), (std, std, std) ))
    return transforms.Compose(transform_list)
"""

def random_crop(img, bbox, crop_size):
    """
    bbox: (x1, y1, x2, y2)
    img: (h, w, 3)
    """
    #ww, hh = img.shape[1], img.shape[0]
    ww = img.width
    hh = img.height
    crop_x, crop_y = np.random.randint(0, ww-crop_size+1), \
                    np.random.randint(0, hh-crop_size+1)
    # Crop the image
    #img_cropped = img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    img_cropped = img.crop((crop_x, crop_y, 
                            crop_x+crop_size, crop_y+crop_size))
    # Fix the bounding box
    bbox_new = (max(0, bbox[0]-crop_x), 
                max(0, bbox[1]-crop_y),
                bbox[2]-crop_x, 
                bbox[3]-crop_y)
    #min(bbox[2], crop_size), 
    #min(bbox[3], crop_size))
    return img_cropped, torch.FloatTensor(bbox_new)

def horizontal_flip(img, bbox):
    """
    bbox: (x1, y1, x2, y2)
    img: (h, w, 3)
    See: https://blog.paperspace.com/data-augmentation-for-bounding-boxes/
    """
    # Create a bbox in the form (x1,y1,x2,y2)
    bbox_np = np.asarray(
        [bbox[0], bbox[1], bbox[2], bbox[3]]).astype(np.float32)
    
    img_shape = (img.height, img.width)
    
    img_center = np.array(img_shape)[::-1]/2
    img_center = np.hstack((img_center, img_center))
    bbox_np[[0,2]] += 2*(img_center[[0,2]] - bbox_np[[0,2]])
    box_w = abs(bbox_np[0] - bbox_np[2])
    bbox_np[0] -= box_w
    bbox_np[2] += box_w
        # Ok now flip the image
    #img_new = img[:,::-1]
    img_new = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img_new, torch.from_numpy(bbox_np)

def resize(img, bbox, size):
    """
    Resize an image and correct the bounding box.

    bbox: (x1, y1, width, height)
    img: (h, w, 3)
    See: https://blog.paperspace.com/data-augmentation-for-bounding-boxes/
    """
    hh, ww = img.height, img.width
    img_resized = F.resize(img, (size, size))
    bbox_new = [
        int(bbox[0] / ww * size),
        int(bbox[1] / hh * size),
        (bbox[2] / ww * size),
        (bbox[3] / hh * size)
    ]
    return img_resized, torch.FloatTensor(bbox_new)

def to_tensor(img):
    return F.to_tensor(img)
