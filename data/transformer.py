from PIL import Image
from torchvision import transforms

def get_transformer(img_size, crop_size, mean, std):
    transform_list = []
    transform_list.append(transforms.Resize(img_size, Image.BICUBIC))
    transform_list.append(transforms.RandomCrop(crop_size))
    transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize( (mean, mean, mean), (std, std, std) ))
    return transforms.Compose(transform_list)
