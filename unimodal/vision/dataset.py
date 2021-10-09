import numpy as np
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
import base64
from PIL import Image
np.random.seed(0)

class ImageDataset(Dataset):
    def __init__(self, cfg, ids, dataset):
        self.cfg = cfg
        self._ids = ids
        self._num_ids = len(ids)
        self._image_paths = []
        self.lineidx_path = cfg.lineidx_path
        self.image_tsv = cfg.image_tsv

        with open(self.lineidx_path, "r") as fp_lineidx:
            self.lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]
        image_paths = []
        for id in self._ids:
            pos_path = []
            neg_path = []
            for f in dataset[id]['img_posFacts']:
                pos_path.append(f['image_id'])
            for f in dataset[id]['img_negFacts']:
                neg_path.append(f['image_id'])
            image_paths.append([pos_path, neg_path])
        self._image_paths = image_paths
        self._num_queries = len(self._image_paths)

    def __len__(self):
        return self._num_queries
    
    def load_img(self, image_id):
        with open(self.image_tsv, "r") as fp:
            fp.seek(self.lineidx[int(image_id)%10000000])
            imgid, img_base64 = fp.readline().strip().split('\t')
        image = cv2.imdecode(np.frombuffer(base64.b64decode(img_base64), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            image = Image.fromarray(image)
            tfs = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
            image = tfs(image)
            return image
        else:
            return None

    def __getitem__(self, idx):
        pos_ids = self._image_paths[idx][0]
        neg_ids = self._image_paths[idx][1]
        pos_imgs = []
        neg_imgs = []
        for image_id in pos_ids:
            pos_imgs.append(self.load_img(image_id))

        for image_id in neg_ids:
            if self.load_img(image_id) != None:
                neg_imgs.append(self.load_img(image_id))    

        return pos_imgs, neg_imgs
        
