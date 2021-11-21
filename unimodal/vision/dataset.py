import numpy as np
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
import base64
from PIL import Image
from transformers import CLIPProcessor
np.random.seed(0)

class ImageDataset(Dataset):
    def __init__(self, cfg, ids, dataset, preprocess):
        self.cfg = cfg
        self._ids = ids
        self._num_ids = len(ids)
        self._image_paths = []
        self.lineidx_path = cfg.lineidx_path
        self.image_tsv = cfg.image_tsv
        #self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.preprocess = preprocess
        with open(self.lineidx_path, "r") as fp_lineidx:
            self.lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]
        image_paths = []
        for id in self._ids:
            for f in dataset[id]['img_posFacts']:
                image_paths.append(f['image_id'])   
            #for f in dataset[id]['img_negFacts']:
            #    image_paths.append(f['image_id'])
        self._image_paths = list(set(image_paths))
        self._num_images = len(self._image_paths)

    def __len__(self):
        #return self._num_ids
        return self._num_images
    
    def __getitem__(self, idx):
        image_id = self._image_paths[idx]
        with open(self.image_tsv, "r") as fp:
            fp.seek(self.lineidx[int(image_id)%10000000])
            imgid, img_base64 = fp.readline().strip().split('\t')
        image = cv2.imdecode(np.frombuffer(base64.b64decode(img_base64), dtype=np.uint8), cv2.IMREAD_COLOR)
        image = image[:,:,::-1]
        image = Image.fromarray(image)
        #inputs = self.processor(images=image, return_tensors="pt", padding=True)
        image = self.preprocess(image)
        return image, image_id
        
