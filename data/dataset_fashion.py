import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from data.transformer import get_transformer

BAD_FILENAMES = [
    "Knit_Bodycon_Skirt/img_00000017.jpg",
    "Striped_Maxi_Dress/img_00000002.jpg",
    "Crinkled_Satin_Halter_Dress/img_00000036.jpg"
]

# NOTE: the string names in both attr and category
# files do align, so we don't need to worry about
# this.

def get_list_attr_img(root):
    filename = "%s/Anno/list_attr_img.txt" % root
    f = open(filename)
    # Skip the first two lines.
    f.readline() # num files
    f.readline() # header
    # Process line-by-line.
    filenames = []
    attrs = []
    for line in f:
        line = line.rstrip().split()
        filename = line[0].replace("img/", "")
        if filename in BAD_FILENAMES:
            continue
        attr = [elem.replace("-1", "0") for elem in line[1::]]
        attr = torch.FloatTensor([float(x) for x in attr])
        filenames.append(filename)
        attrs.append(attr)
    f.close()
    return filenames, attrs

def get_list_category_img(root):
    filename = "%s/Anno/list_category_img.txt" % root
    f = open(filename)
    # Skip the first two lines.
    num_files = int(f.readline())
    f.readline()
    categories = []
    # Process line-by-line.
    for line in f:
        line = line.rstrip().split()
        filename = line[0].replace("img/", "")
        if filename in BAD_FILENAMES:
            continue
        category = int(line[-1])
        categories.append(category)
    f.close()
    return categories


class DeepFashionDataset(Dataset):
    def __init__(self,
                 root,
                 filenames,
                 indices,
                 attrs,
                 categories,
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
        assert len(filenames) == len(attrs) == len(categories)
        # self.transform = transforms.Compose(transforms_)
        self.root = root
        self.indices = indices
        # Store information about the dataset.
        self.filenames = filenames
        self.attrs = attrs
        self.categories = categories
        self.transformer = get_transformer(img_size=img_size,
                                           crop_size=crop_size,
                                           mean=mean,
                                           std=std)

    def __getitem__(self, index):
        this_idx = self.indices[index]
        filepath = "%s/DF_Img/img/%s" % (self.root, self.filenames[this_idx])
        img = Image.open(filepath)
        img = img.convert("RGB")
        img = self.transformer(img)
        attr_label = self.attrs[this_idx]
        category_label = self.categories[this_idx]
        return img, category_label, attr_label

    def __len__(self):
        return len(self.indices)
