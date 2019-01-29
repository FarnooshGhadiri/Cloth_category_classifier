import torch
from options.options import Options
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from transformer import get_transformer
import piexif
import imghdr
Image.MAX_IMAGE_PIXELS = None

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
        filepath = "%s/High_res/Img/img/%s" % (self.root, self.filenames[int(index)])
        img_type = imghdr.what(filepath)
        try:
            open_img = Image.open(filepath)
        except:
            print('can not open the image')
            print(filepath)
        try:
            tmp_img = self.transformer(open_img)
        except:
            print('can not transfer the image')
            print(filepath)
        #if img_type == 'jpeg':
         # piexif.remove(filepath)
        #try:
         #  tmp_img = self.transformer(Image.open(filepath))
        #except:
         #  print(filepath)
        img=tmp_img[0:3]
        attr_label = self.attrs[int(index)]
        category_label = self.categories[int(index)]
        return img, category_label, attr_label

    def __len__(self):
        return self.num_files


# if __name__ == '__main__':
#  op = Options()
#  opt = op.parse()
#  from torch.utils.data import DataLoader
# #root = os.environ["DEEPFASHION_FOLDER"]
# #root="/home/farnoosh/Projects/DeepFashion/Category and Attribute Prediction Benchmark"
# #train_transforms = [
#  #   transforms.ToTensor(),
#   #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# #]
#  ds = DeepFashionDataset(opt)
#
#  num_train = len(ds)
#  indices = list(range(num_train))
#  print(opt.ratio[1])
#  split =int((opt.ratio[1]+opt.ratio[2])*num_train)
#
#  validation_Test_idx = np.random.choice(indices, size=split, replace=False)
#  train_idx = list(set(indices) - set(validation_Test_idx))
#  train_sampler = SubsetRandomSampler(train_idx)
# #validation Set
#  split = int(round(0.5 * len(validation_Test_idx)))
#  validation_idx = np.random.choice(validation_Test_idx, size=split, replace=False)
#  validation_sampler = SubsetRandomSampler(validation_idx)
# #Test set
#  test_idx=list(set(validation_Test_idx) - set(validation_idx))
#  test_sampler = SubsetRandomSampler(test_idx)
#
#
#  train_loader = DataLoader(ds, batch_size=10, shuffle=False, sampler=train_sampler)
#  Validate_loader = DataLoader(ds, batch_size=10, shuffle=False, sampler=validation_sampler)
#  Test_loader = DataLoader(ds, batch_size=10, shuffle=False, sampler=test_sampler)
#
#  for i, data in enumerate(train_loader):
#     inputs, target_1,target_2  = data
#     inputs_var = Variable(inputs, requires_grad=True)
#     model = load_model(opt, [50,1000])
#     model.train()
#     output = model(inputs_var)
#     print(output)


   #stuff = iter(train_loader).next()
