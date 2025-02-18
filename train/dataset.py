import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# 참고
class T2ICompBench(Dataset):
    def __init__(self, 
                 data_dir, 
                 img_dir,
                 category_list, 
                 split=None):

        self.data = []
        split = "" if split is None else "_" + split
        for category in category_list:
            idx = 0
            with open(os.path.join(data_dir, f"{category}{split}.txt"), 'r') as fin:
                lines = fin.read().splitlines()
                for line in lines:
                    try:
                        img_path = os.path.join(img_dir, f'{line}.png')
                        image=Image.open(img_path)
                        self.data.append({'category': category, 
                                        'caption': line, 
                                        'image': image, 
                                        'idx': idx})
                        idx += 1
                    except Exception as e:
                        print(e)
                        continue
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def collate_fn(self, batch):
        categories = [sample['category'] for sample in batch]
        captions = [sample['caption'] for sample in batch]
        images = [sample['image'] for sample in batch]
        idxs = [sample['idx'] for sample in batch]
        
        return categories, captions, images, idxs



class PairDataset(Dataset):
    def __init__(self, config, transform, tokenizer): # config = data_config
        super().__init__() 
        print("This is for T2ICompBench-train dataset.")
        
        # load data
        self.dataset = []
        self.img_path = config.img_path
        self.config = config
        self.transform = transform
        self.tokenizer = tokenizer

        # config.data_path, config.img_path 로 변경
        with open(config.data_path, 'r') as f:
            lines = f.read().splitlines()
            for line_idx, line in enumerate(lines):
                sample = {
                    "idx": line_idx,
                    "prompt": line,
                    "image": f"{line}.png",
                }
                self.dataset.append(sample)
    
    def bsz_fn(self):
        return self.config.batch_size 
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        try:
            img_path = os.path.join(self.img_path, sample["image"])
            with Image.open(img_path) as img:
                image = img.convert('RGB')
                image = self.transform(image)
                yield image, sample["prompt"], sample["idx"]
        except Exception as e:
            print(e)

    def __iter__(self):
        for sample in self.dataset:
            try:
                # Load Image
                # prompt = sample["prompt"]
                img_path = os.path.join(self.img_path, sample["image"])
                with Image.open(img_path) as img:
                    image = img.convert('RGB')
                    image = self.transform(image)
                    yield image, sample["prompt"], sample["idx"]

            except Exception as e:
                print(e)
                continue

    def collate_fn(self, batch):
        images, captions, idxs = zip(*batch)
        return list(images), list(captions), list(idxs)

