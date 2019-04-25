import os

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def opt2file(opt, dst_file):
    args = vars(opt) 
    with open(dst_file, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        print ('------------ Options -------------')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
            print ("%s: %s" %(str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
        print ('-------------- End ----------------')
