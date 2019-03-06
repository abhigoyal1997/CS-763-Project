import os
import json

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class BinaryQADataset(Dataset):
    def __init__(self, root, image_size, size=None, split='train'):
        super(Dataset, self).__init__()
        self.image_size = image_size
        self.post_transform = transforms.ToTensor()
        self.image_dir = os.path.join(root, 'images')
        self.split = split

        with open(os.path.join(root, 'questions.json'),'r') as f:
            qjson = json.load(f)
        qjson = qjson['questions']

        questions = {}
        for q in qjson:
            questions[q['question_id']] = (q['image_id'],q['question'])

        with open(os.path.join(root, 'annotations.json'),'r') as f:
            ajson = json.load(f)
        ajson = ajson['annotations']

        self.instances = []
        for a in ajson:
            self.instances.append((
                a['image_id'],
                questions[a['question_id']][1],
                [(ai['answer'][0] == 'y') for ai in a['answers']]
            ))

        print(f'Found {len(self.instances)} instances!')

    def transformation(self, imgs):
        for i in range(len(imgs)):
            if imgs[i].size != self.image_size:
                imgs[i] = transforms.functional.resize(imgs[i], self.image_size)

        if self.post_transform is not None:
            for i in range(len(imgs)):
                imgs[i] = self.post_transform(imgs[i])
        return imgs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x = Image.open(os.path.join(self.image_dir, f'abstract_v002_{self.split}2015_{self.instances[index][0]:012}.png'))
        x = self.transformation([x])[0]
        q = self.instances[index][1]
        y = self.instances[index][2]

        return [x,q,y]
