import torch
from options.options import Options
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from data.transformer import get_transformer
import piexif
import imghdr
Image.MAX_IMAGE_PIXELS = None

class DeepFashionDataset(Dataset):
    def __init__(self, opt):
        """
        Parameters
        ----------
        root: the root of the DeepFashion dataset. This is the folder
          which contains the subdirectories 'Anno', 'High_res', etc. 
          
        """
        # self.transform = transforms.Compose(transforms_)
        self.root = opt.dir
        self.Ctg_num=opt.numctg
        # Store information about the dataset.
        self.filenames = None
        self.attrs = None
        self.categories = None
        self.num_files = None
        # Read the metadata files.
        self.get_list_attr_img()
        self.get_list_category_img()
        self.transformer = get_transformer(opt)

    def get_list_attr_img(self):
        filename = "%s/Anno/list_attr_img.txt" % self.root
        f = open(filename)
        # Skip the first two lines.
        num_files = int(f.readline())
        self.num_files = num_files
        self.filenames = [None] * num_files
        self.attrs = [None] * num_files
        f.readline()
        # Process line-by-line.
        i = 0
        for line in f:
                line = line.rstrip().split()
                filename = line[0].replace("img/", "")
                attr = [elem.replace("-1", "0") for elem in line[1::]]
                attr = torch.FloatTensor([float(x) for x in attr])
                self.filenames[i] = filename
                self.attrs[i] = attr
                i = i+1
        f.close()

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
        filepath = "%s/High_res/Img/img/%s" % (self.root, self.filenames[int(index)])
        img_type = imghdr.what(filepath)
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
        attr_label = self.attrs[int(index)]
        category_label = self.categories[int(index)]
        return img, category_label, attr_label

    def __len__(self):
        return self.num_files


