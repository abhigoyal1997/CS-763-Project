import os
import json
import pandas as pd
import torch
import numpy as np

from collections import Counter
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class BinaryQADataset(Dataset):
    def __init__(self, root, image_size, nlp, size=None, split='train', maxlen=None):
        super(Dataset, self).__init__()
        self.image_size = image_size
        self.post_transform = transforms.ToTensor()
        self.image_dir = os.path.join(root, 'images')
        self.split = split
        self.nlp = nlp

        self.df = self.text_pre_process(root)

        if maxlen is None:
            self.maxlen = self.df.length.values.max()
        else:
            self.maxlen = maxlen
            self.df['length'] = self.df.length.apply(lambda x: min(self.maxlen, x))

        self.df['q_pad'] = self.df.q_idx.apply(self.pad_text)

        print(f'Found {len(self.df)} instances!')

    def indexer(self, s):
        return [self.word2idx[w.text.lower()] for w in self.nlp(s)]

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
        x = Image.open(os.path.join(self.image_dir, f'abstract_v002_{self.split}2015_{self.df.image_id[index]:012}.png'))
        x = self.transformation([x])[0]
        q = self.df.q_pad[index]
        y = torch.Tensor(self.df.answers[index])
        length = self.df.length[index]

        return [x,q,y,length]

    def pad_text(self, s):
        padded = np.zeros((self.maxlen,), dtype=np.int64)
        if len(s) > self.maxlen:
            padded[:] = s[:self.maxlen]
        else:
            padded[:len(s)] = s
        return padded

    def text_pre_process(self, root):
        with open(os.path.join(root, 'questions.json'),'r') as f:
            q_df = pd.DataFrame(json.load(f)['questions'])

        q_df['question'] = q_df.question.apply(lambda x:x.strip())

        words = Counter()
        for q in q_df.question.values:
            words.update(w.text.lower() for w in self.nlp(q))

        words = sorted(words, key=words.get, reverse=True)
        words = ['_PAD','_UNK'] + words

        self.word2idx = {o:i for i,o in enumerate(words)}
        self.idx2word = {i:o for i,o in enumerate(words)}

        q_df['q_idx'] = q_df.question.apply(self.indexer)
        q_df.set_index('question_id', inplace=True)

        with open(os.path.join(root, 'annotations.json'),'r') as f:
            a_df = pd.DataFrame(json.load(f)['annotations'])

        a_df['answers'] = a_df.answers.apply(lambda x: [int(xi['answer'][0]=='y') for xi in x])
        a_df.set_index('question_id', inplace=True)

        df = a_df.merge(q_df, how='inner')[['question', 'q_idx', 'answers','image_id']]
        df['length'] = df.q_idx.apply(len)

        return df
