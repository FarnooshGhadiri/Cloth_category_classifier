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

class DeepFashionDataset(Dataset):
    def __init__(self,
                 root,
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
        # Store information about the dataset.
        self.filenames = None
        self.attrs = None
        self.categories = None
        # Read the metadata files.
        self.get_list_attr_img()
        self.get_list_category_img()
        self.transformer = get_transformer(img_size=img_size,
                                           crop_size=crop_size,
                                           mean=mean,
                                           std=std)

    def get_list_attr_img(self):
        filename = "%s/Anno/list_attr_img.txt" % self.root
        f = open(filename)
        # Skip the first two lines.
        num_files = int(f.readline())
        #self.filenames = [None] * num_files
        #self.attrs = [None] * num_files
        f.readline()
        # Process line-by-line.
        filenames = []
        attrs = []
        for line in f:
            line = line.rstrip().split()
            filename = line[0].replace("img/", "")
            if filename not in BAD_FILENAMES:
                attr = [elem.replace("-1", "0") for elem in line[1::]]
                attr = torch.FloatTensor([float(x) for x in attr])
                filenames.append(filename)
                attrs.append(attr)
            #self.filenames[i] = filename
            #self.attrs[i] = attr
        f.close()
        self.filenames = filenames
        self.attrs = attrs

    def get_list_category_img(self):
        filename = "%s/Anno/list_category_img.txt" % self.root
        f = open(filename)
        # Skip the first two lines.
        num_files = int(f.readline())
        self.categories = [None] * num_files
        f.readline()
        # Process line-by-line.
        i = 0
        for line in f:
            line = line.rstrip().split()
            filename = line[0].replace("img/", "")
            category = int(line[-1])
            self.categories[i] = category
            i = i+1
        f.close()

    def __getitem__(self, index):
        filepath = "%s/DF_Img/img/%s" % (self.root, self.filenames[index])
        #img_type = imghdr.what(filepath)
        """
        try:
            open_img = Image.open(filepath)
            open_img = open_img.convert("RGB")
        except:
            print('can not open the image')
            print(filepath)
        try:
            tmp_img = self.transformer(open_img)
        except:
            print('can not transfer the image')
            print(filepath)
        """
        img = Image.open(filepath)
        img = img.convert("RGB")
        img = self.transformer(img)
        attr_label = self.attrs[int(index)]
        category_label = self.categories[int(index)]
        return img, category_label, attr_label

    def __len__(self):
        return len(self.filenames)
