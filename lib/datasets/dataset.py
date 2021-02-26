import os.path as osp
from glob import glob
import re

class DataSet(object):
    def __init__(self, data_dir, name='veri776', info=True):
        self.name = name
        self.images_dir = osp.join(data_dir, name)
        self.train_path = 'bounding_box_train'
        #self.train_path = 'train'
        self.query_path = 'query' # for test-10000 dataset (large)
        self.gallery_path = 'bounding_box_test' #if u use 'veri-wild dataset, it's automatically defined by test-10000 protocol'
        if self.name=='veri-wild':
            self.train_path = 'train_set'
            self.small_query_path = 'query_small'
            self.small_gallery_path = 'test_small'
            self.middle_query_path = 'query_middle'
            self.middle_gallery_path = 'test_middle'
            
            self.query_small, self.gallery_small, self.query_middle, self.gallery_middle = [],[],[],[]
            self.num_query_small_ids, self.num_gallery_small_ids,  self.num_query_middle_ids, self.num_gallery_middle_ids = 0, 0, 0, 0

        self.train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0

        self.cam_dict = self.set_cam_dict()
        self.num_cam = self.cam_dict[name]

        self.load(info)

    def set_cam_dict(self):
        cam_dict = {}
        cam_dict['veri776'] = 18
        cam_dict['veri-wild'] = 174
        cam_dict['vehicleid'] = 12
        return cam_dict

    def preprocess(self, images_dir, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        all_pids = {}
        idx2pid = []
        ret = []
        _tmp_list = glob(osp.join(images_dir, path, '*.jpg'))
        #if path=='query':
        #    _tmp_list.append('data/duke/query/0005_c6_f0030883.jpg')
        fpaths = sorted(_tmp_list)
        cnt = 0
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: 
                continue  # junk images are just ignored
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            ret.append((fname, pid, cam, cnt))
            idx2pid.append(pid)
            cnt = cnt + 1
        if relabel:
            return ret, int(len(all_pids)), idx2pid
        else:
            return ret, int(len(all_pids))

    def load(self, info=True):
        self.train, self.num_train_ids, self.idx2pid = self.preprocess(self.images_dir, self.train_path)
        self.query, self.num_query_ids = self.preprocess(self.images_dir, self.query_path, relabel=False)
        self.gallery, self.num_gallery_ids = self.preprocess(self.images_dir, self.gallery_path, relabel=False)

        if self.name=='veri-wild':
            self.small_query, self.num_query_small_ids = self.preprocess(self.images_dir,self.small_query_path,relabel=False)
            self.small_gallery, self.num_gallery_small_ids = self.preprocess(self.images_dir,self.small_gallery_path,relabel=False)
            self.middle_query, self.num_query_middle_ids = self.preprocess(self.images_dir,self.middle_query_path,relabel=False)
            self.middle_gallery, self.num_gallery_middle_ids = self.preprocess(self.images_dir,self.middle_gallery_path,relabel=False)

        if info:
            if self.name=='veri-wild':
                print(self.__class__.__name__, self.name, "loaded")
                print("  subset   | # ids | # images")
                print("  ---------------------------")
                print("        veri-wild train      ")
                print("  ---------------------------")
                print("  train    | 'Unknown' | {:8d}"
                    .format(len(self.train)))
                print("  ---------------------------")
                print("        veri-wild large      ")
                print("  ---------------------------")
                print("  query    | {:5d} | {:8d}"
                    .format(self.num_query_ids, len(self.query)))
                print("  gallery  | {:5d} | {:8d}"
                    .format(self.num_gallery_ids, len(self.gallery)))
                print("  ---------------------------")
                print("       veri-wild middle      ")
                print("  ---------------------------")
                print("  query    | {:5d} | {:8d}"
                    .format(self.num_query_middle_ids, len(self.middle_query)))
                print("  gallery  | {:5d} | {:8d}"
                    .format(self.num_gallery_middle_ids, len(self.middle_gallery)))
                print("  ---------------------------")
                print("        veri-wild small      ")
                print("  ---------------------------")
                print("  query    | {:5d} | {:8d}"
                    .format(self.num_query_small_ids, len(self.small_query)))
                print("  gallery  | {:5d} | {:8d}"
                    .format(self.num_gallery_small_ids, len(self.small_gallery)))

            else:    
                print(self.__class__.__name__, self.name, "loaded")
                print("  subset   | # ids | # images")
                print("  ---------------------------")
                print("  train    | 'Unknown' | {:8d}"
                    .format(len(self.train)))
                print("  query    | {:5d} | {:8d}"
                    .format(self.num_query_ids, len(self.query)))
                print("  gallery  | {:5d} | {:8d}"
                    .format(self.num_gallery_ids, len(self.gallery)))
