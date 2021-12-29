# encoding=utf-8
import os
import numpy as np
import torch


class Dataset:

    def __init__(self, filename, seg, word2id, querylen, targetlen):
        self.query_max_len = querylen
        self.target_max_len = targetlen
        query, content, translation, annotate, analyze, labels = [], [], [], [], [], []
        if os.path.exists(filename):    # 训练数据的路径
            with open(filename, 'r', encoding='utf-8') as fe:
                for line in fe:
                    line = line.strip().split(',')
                    # q, t, label = line[1], line[2], line[-1]    # 从txt文件中读数据，需要符合特定的格式
                    length = len(line)
                    q, a, b, c, d, label = line[0], line[1], line[2], line[3], line[4], line[5]     # query，原文，译文，注释,赏析，标签
                    # 都转换成word2vec中的id
                    query.append([word2id.get(qw, word2id['UNK']) for qw in seg(q, ifremove=False)['tokens']])
                    content.append([word2id.get(aw, word2id['UNK']) for aw in seg(a, ifremove=False)['tokens']])
                    translation.append([word2id.get(bw, word2id['UNK']) for bw in seg(b, ifremove=False)['tokens']])
                    annotate.append([word2id.get(cw, word2id['UNK']) for cw in seg(c, ifremove=False)['tokens']])
                    analyze.append([word2id.get(dw, word2id['UNK']) for dw in seg(d, ifremove=False)['tokens']])
                    labels.append(1) if label == '1' else labels.append(0)
                # 译文和注释要进行拼接
                self.query = query
                self.content = content
                self.translation = translation.extend(annotate)
                self.analyze = analyze
                self.labels = labels
        else:
            raise FileNotFoundError(f"train data file not found in {filename}")

    def batch(self, index):
        query_ids, content_ids, tran_ids, analyze_ids, label_ids = self.query[index], self.content[index], \
                                                                   self.translation[index], self.analyze[index], self.labels[index]
        s1_data, s1_mask = self.pad2longest(s1_ids, self.query_max_len)
        s2_data, s2_mask = self.pad2longest(s2_ids, self.target_max_len)
        return s1_data, s2_data, s1_mask, s2_mask, s_labels

    def __getitem__(self, index):
        s1_ids, s2_ids, s_labels = self.query[index], self.target[index], self.labels[index]
        s1_data, s1_mask = self.pad2longest(s1_ids, self.query_max_len)
        s2_data, s2_mask = self.pad2longest(s2_ids, self.target_max_len)
        # return torch.LongTensor(s1_data).to(self.config['device']), torch.LongTensor(s2_data).to(self.config['device']), torch.FloatTensor(s1_mask).to(self.config['device']), torch.FloatTensor(s2_mask).to(self.config['device']), torch.LongTensor([s_labels]).to(self.config['device'])
        return torch.LongTensor(s1_data), torch.LongTensor(s2_data), torch.FloatTensor(s1_mask), torch.FloatTensor(s2_mask), torch.LongTensor([s_labels])

    @staticmethod
    def pad2longest(data_ids, max_len):
        if isinstance(data_ids[0], list):   # 统一长度，超过的截断，没有超过的用0补齐
            s_data = np.array([s[:max_len] + [0] * (max_len - len(s[:max_len])) for s in data_ids])
            s_mask = np.array([[1] * m[:max_len] + [0] * (max_len - len(m[:max_len])) for m in data_ids])
        elif isinstance(data_ids[0], int):
            s_data = np.array(data_ids[:max_len] + [0] * (max_len - len(data_ids[:max_len])))
            s_mask = np.array([1] * len(data_ids[:max_len]) + [0] * (max_len - len(data_ids[:max_len])))
        else:
            raise TypeError("list type is required")
        return (s_data, s_mask)

    def cutAnalyze(self):   # 给赏析进行分段

        return



    def __len__(self):
        return len(self.labels)

