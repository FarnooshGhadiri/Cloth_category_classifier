import copy
from PIL import Image
from torchvision import transforms


def get_transformer(opt):
    transform_list = []
    
    # resize  
    osize = [opt.load_size, opt.load_size]
    transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    

    # flip
    if opt.mode == "Train" and opt.flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    # to tensor
    transform_list.append(transforms.ToTensor())
    
    # If you make changes here, you should also modified 
    # function `tensor2im` in util/util.py accordingly
    transform_list.append(transforms.Normalize(opt.mean, opt.std))

    return transforms.Compose(transform_list)


