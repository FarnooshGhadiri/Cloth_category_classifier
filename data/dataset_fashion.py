import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
#from data.transformer import get_transformer
from data.transformer import (random_crop,
                              horizontal_flip,
                              resize,
                              to_tensor)

BAD_FILENAMES = [
    "Knit_Bodycon_Skirt/img_00000017.jpg",
    "Striped_Maxi_Dress/img_00000002.jpg",
    "Crinkled_Satin_Halter_Dress/img_00000036.jpg"
]

# NOTE: the string names in both attr and category
# files do align, so we don't need to worry about
# this.

def get_list_attr_img(root, max_lines=-1):
    filename = "%s/Anno/list_attr_img.txt" % root
    f = open(filename)
    # Skip the first two lines.
    f.readline() # num files
    f.readline() # header
    # Process line-by-line.
    dd = dict()
    for i, line in enumerate(f):
        line = line.rstrip().split()
        filename = line[0].replace("img/", "")
        if filename in BAD_FILENAMES:
            continue
        attr = [elem.replace("-1", "0") for elem in line[1::]]
        attr = torch.FloatTensor([float(x) for x in attr])
        dd[filename] = attr
        if i == max_lines:
            break
    f.close()
    return dd

def get_list_category_img(root, max_lines=-1):
    filename = "%s/Anno/list_category_img.txt" % root
    f = open(filename)
    # Skip the first two lines.
    num_files = int(f.readline())
    f.readline()
    dd = dict()
    # Process line-by-line.
    for i, line in enumerate(f):
        line = line.rstrip().split()
        filename = line[0].replace("img/", "")
        if filename in BAD_FILENAMES:
            continue
        # BUG: The label is meant to be zero-indexed, but
        # it looks like I forgot to do this. This means
        # that at test time, when you grab the predicted
        # label using argmax(), subtract 1 so that it is
        # now zero-indexed.
        category = int(line[-1]) # should be int(line[-1])-1
        dd[filename] = category
        if i == max_lines:
            break
    f.close()
    return dd

def get_attr_names_and_types(root, max_lines=-1):
    filename = "%s/Anno/list_attr_cloth.txt" % root
    f = open(filename)
    num_files = int(f.readline())
    f.readline()
    attrs_name = []
    attrs_type = []
    for i, line in enumerate(f):
        word = line.strip()[:-1].strip()
        word2 = line.strip()[-1]
        attrs_name.append(word)
        attrs_type.append(int(word2))
        if i == max_lines:
            break
    f.close()
    return attrs_name, attrs_type

def get_weight_attr_img(root):
     filename = "%s/Anno/list_attr_img.txt" % root
     f = open(filename)
     # Skip the first two lines.
     num_files = int(f.readline())
     f.readline()
     # Process line-by-line.
     i = 0
     sum_attr = torch.zeros(1000)
     for line in f:
        line = line.rstrip().split()
        filename = line[0].replace("img/", "")
        attr = [elem.replace("-1", "0") for elem in line[1::]]
        attr = torch.FloatTensor([float(x) for x in attr])
        sum_attr += attr
        i = i+1
     f.close()
     weight_attr = 1-(sum_attr/(sum_attr.sum()))
     return weight_attr

def get_bboxes(root):
    dd = {}
    with open("%s/Anno/list_bbox.txt" % root) as f:
        for line in f:
            line = line.rstrip().split()
            filename = line[0].replace("img/", "")
            # in the form [x1, y1, x2, y2]
            bbox = [ int(x) for x in line[1:] ]
            #bbox[2] = bbox[2] - bbox[0]
            #bbox[3] = bbox[3] - bbox[1]
            bbox = torch.FloatTensor(bbox)
            dd[filename] = bbox
    return dd
 
class DeepFashionDataset(Dataset):
    def __init__(self,
                 root,
                 indices,
                 attrs,
                 categories,
                 bboxes,
                 data_aug=False,
                 img_size=256,
                 crop_size=224,
                 mean=0.5,
                 std=0.5):
        """
        Parameters
        ----------
        root: the root of the DeepFashion dataset. This is the folder
          which contains the subdirectories 'Anno', 'High_res', etc.
          
        """
        super(DeepFashionDataset, self).__init__()
        # self.transform = transforms.Compose(transforms_)
        self.root = root
        self.indices = indices
        # Store information about the dataset.
        self.filenames = list(attrs.keys())
        self.attrs = attrs
        self.categories = categories
        self.bboxes = bboxes
        for arr in [attrs, categories, bboxes]:
            print("length: ", len(arr))
        self.data_aug = data_aug
        self.crop_size = crop_size
        self.img_size = img_size
        self.mean = mean
        self.std = std
        
        #self.transformer = get_transformer(img_size=img_size,
        #                                   crop_size=crop_size,
        #                                   mean=mean,
        #                                   std=std)

    def __getitem__(self, index):
        this_filename = self.filenames[index]
        
        filepath = "%s/Img/img/%s" % (self.root, this_filename)
        img = Image.open(filepath)
        img = img.convert("RGB")
        
        if not self.data_aug:
            # Resize the image to the crop size
            bbox_label = self.bboxes[this_filename].clone()
            img, bbox_label = resize(img, bbox_label, self.crop_size)
            # Bring the bounding boxes to be in [0,1]
            bbox_label[0] /= self.crop_size
            bbox_label[2] /= self.crop_size
            bbox_label[1] /= self.crop_size
            bbox_label[3] /= self.crop_size
        else:
            bbox_label = self.bboxes[this_filename].clone()
            # Resize the image to the image size
            img, bbox_label = resize(img, bbox_label, self.img_size)
            # Randomly crop an image of size crop_size
            img, bbox_label = random_crop(img, bbox_label, self.crop_size)

            # With 0.5 probability, horizontally flip
            # the image
            if torch.rand(1).item() < 0.5:
                img, bbox_label = horizontal_flip(img, bbox_label)
                
            # Bring the bounding boxes to be in [0,1]
            bbox_label[0] /= self.crop_size
            bbox_label[2] /= self.crop_size
            bbox_label[1] /= self.crop_size
            bbox_label[3] /= self.crop_size

        img = (to_tensor(img) - self.mean) / self.std

        #img = self.transformer(img)
        
        attr_label = self.attrs[this_filename]
        category_label = self.categories[this_filename]
        
        return filepath, img, category_label, attr_label, bbox_label

    def __len__(self):
        return len(self.indices)
