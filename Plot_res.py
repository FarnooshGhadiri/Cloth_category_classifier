import matplotlib.pyplot as plt
import numpy as np
import torch
root="/home/farnoosh/Projects/DeepFashion/Category and Attribute Prediction Benchmark"
from data.input import get_attr_name

filename = "/home/farnoosh/Projects/DeepFashion/Category and Attribute Prediction Benchmark/Anno/list_attr_img.txt"
f = open(filename)
# Skip the first two lines.
num_files = int(f.readline())
filenames = [None] * num_files
attrs = [None] * num_files
f.readline()
# # Process line-by-line.
i = 0
sum_attr = torch.zeros(1000,dtype=torch.float)
# for line in f:
#    line = line.rstrip().split()
#    filename = line[0].replace("img/", "")
#    attr = [elem.replace("-1", "0") for elem in line[1::]]
#    attr = torch.FloatTensor([float(x) for x in attr])
#    filenames[i] = filename
#    attrs[i] = attr
#    sum_attr += attr
#    i = i + 1
# f.close()
#
# f = open('Sum_Attr.txt','a')
#
# for i in range(len(sum_attr)):
#    f.write("%f  " % (sum_attr[i]))
# f.close()


f = open('Sum_Attr.txt')
line = f.readline()
tmp = line.strip().split()
sum_attr = torch.FloatTensor([float(x) for x in tmp])




f = open("Test_Acc_Per_Attr.txt")
f.readline()
f.readline()
f.readline()
line=f.readline()
line=line.strip().split()
attr = [elem.replace('nan', '0') for elem in line]
attr = torch.FloatTensor([float(x) for x in attr])
# attrs_name, attrs_type = get_attr_name(root)
# y_pos = np.arange(len(attr))
# plt.figure(figsize=(150, 50))
# bars = plt.bar(y_pos, attr, align='edge', alpha=0.8, width=0.8)# #sns.barplot(y_pos, performance)
# plt.xticks(y_pos, attrs_name,rotation=90)
# plt.ylabel('# Top 5 Accuracy')
# #plt.show()
# plt.savefig('Top5accuracy',bbox_inches='tight')


plt.figure(figsize=(20, 10))
y_pos=torch.log(sum_attr)
bars = plt.scatter(y_pos, attr)# #sns.barplot(y_pos, performance)
plt.xlabel('log(#Images)')
#plt.xticks(y_pos, attrs_name,rotation=90)
plt.ylabel('Top 5 Recall')
#plt.show()
plt.savefig('Num_img_recall_1000',bbox_inches='tight')
