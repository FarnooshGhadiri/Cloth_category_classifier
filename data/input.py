from options.options import Options
import torch


def get_attr_name(opt):
    filename = "%s/Anno/list_attr_cloth.txt" % opt.dir
    f = open(filename)
    num_files = int(f.readline())
    f.readline()
    attrs_name = []
    attrs_type = torch.zeros(num_files)
    i=0
    for line in f:
       word = line.strip()[:-1].strip()
       word2=line.strip()[-1]
       attrs_name.append(word)
       attrs_type[i] = float(word2)
       i=i+1
    f.close()
    return attrs_name, attrs_type

def get_Ctg_name(opt):
    filename = "%s/Anno/list_category_cloth.txt" % opt.dir
    f = open(filename)
    f.readline()
    f.readline()
    Ctg_name=[]
    Ctg_type=[]
    for line in f:
       word = line.strip()[:-1].strip()
       word2 = line.strip()[-1]
       Ctg_name.append(word)
       Ctg_type.append(word2)

    f.close()
    return Ctg_name, Ctg_type

def get_weight_attr_img(opt):
     filename = "%s/Anno/list_attr_img.txt" % opt.dir
     f = open(filename)
     # Skip the first two lines.
     num_files = int(f.readline())
     #num_files = 50
     f.readline()
     # Process line-by-line.
     i = 0
     sum_attr=torch.zeros(1000)
     for line in f:
      # if i<50:
        line = line.rstrip().split()
        filename = line[0].replace("img/", "")
        attr = [elem.replace("-1", "0") for elem in line[1::]]
        attr = torch.FloatTensor([float(x) for x in attr])
        sum_attr+=attr
        i = i+1
     f.close()
     weight_attr = 1-(sum_attr/max(sum_attr))
     #print(weight_attr.max())
     #print(weight_attr.min())
     return weight_attr

if __name__ == "__main__":
   op = Options()
   opt = op.parse()
   attr_name,attr_type = get_attr_name(opt)
