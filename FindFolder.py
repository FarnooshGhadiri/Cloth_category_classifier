import torch
from options.options import Options
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from data.transformer import get_transformer
from torch.utils.data.sampler import SubsetRandomSampler
import piexif
import imghdr
class DeepFashionDataset(Dataset):
    def __init__(self, opt):
        """
        Parameters
        ----------
        root: the root of the DeepFashion dataset. This is the folder
          which contains the subdirectories 'Anno', 'Img', etc.
          It is assumed that in 'Img' the directory 'img_converted'
          exists, which gets created by running the script `resize.sh`.
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
        filepath = "%s/Img/img/%s" % (self.root, self.filenames[index])
        img_type=imghdr.what(filepath)
        if img_type=='jpeg':
            piexif.remove(filepath)
            try:
              tmp_img1=Image.open(filepath)
            except:
               print('after piexif...')
               print(filepath)
           # img=tmp_img[0:3]
           # print(img.size) 
            tmp_img = self.transformer(tmp_img1)
            img=tmp_img[0:3]
           # print(img.size())
        else:
           # print(img.size())
            #print(filepath)
            tmp_img1=Image.open(filepath)
            #img=img.convert('RGB')
           # img=tmp_img[0:3]
           # print('---PNG')
           # print(img.size)
            tmp_img= self.transformer(tmp_img1)
            img=tmp_img[0:3]
           # print(img.size())
        attr_label = self.attrs[index]
        category_label = self.categories[index]
        return img, category_label, attr_label

    def __len__(self):
        return self.num_files


if __name__ == '__main__':
  op = Options()
  opt = op.parse()
  from torch.utils.data import DataLoader
  train_transforms = [
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ]
  ds = DeepFashionDataset(opt)

  num_train = len(ds)
  indices = list(range(num_train))
  train_idx=SubsetRandomSampler(indices)
  train_loader = DataLoader(ds, batch_size=10, shuffle=False, sampler=train_idx)
  for i, data in enumerate(train_loader):
     inputs, target_1,target_2  = data
