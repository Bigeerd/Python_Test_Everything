
def merge_respective_part(data):
    #run:test_merge_respective_part()
    import torch
    part1 = torch.stack([_[0] for _ in data], dim=0)
    part2 = torch.stack([_[1] for _ in data], dim=0)
    part3 = torch.cat([_[2] for _ in data], dim=0)
    part4 = data[0][3]
    return part1, part2, part3, part4

# def test_merge_respective_part():
#     import torch
#     fake_data = [
#         # 每条格式：[正样本, 负样本, 权重, ..., mode]
#         [torch.tensor([10, 10, 10]), torch.tensor([11, 11, 11]), torch.tensor([1.2]), "tail-batch"],
#         [torch.tensor([20, 20, 20]), torch.tensor([21, 21, 21]), torch.tensor([2.2]), "tail-batch"],
#     ]
#     a,b,c,d = merge_respective_part(fake_data)
#     print(f'a:{a}, b:{b}, c:{c}, d:{d}')

def count_frequency(triples, start=4):
    #run test_count_frequency()
    count = {}
    for h, r, t in triples:
        tmp1,tmp2 = count.get((h,r), None),count.get((t,-r-1), None)
        count[(h,r)] = start if tmp1 is None else tmp1 + 1#这是人家自定义的规则
        count[(t,-r-1)] = start if tmp2 is None else tmp2 + 1#这是人家自定义的规则
    return count

# def test_count_frequency():
#     test_triples = [
#         [0, 1, 2],
#         [0, 1, 3],
#         [4, 5, 6],
#     ]
#     print(count_frequency(test_triples))

def get_true_head_and_tail(triples):
    #run test_get_true_head_and_tail()
    import numpy as np

    true_head = {}
    true_tail = {}

    for head, relation, tail in triples:
        if (head, relation) not in true_tail:
            true_tail[(head, relation)] = []
        true_tail[(head, relation)].append(tail)
        if (relation, tail) not in true_head:
            true_head[(relation, tail)] = []
        true_head[(relation, tail)].append(head)

    for relation, tail in true_head:
        true_head[(relation, tail)] = np.array(
            list(set(true_head[(relation, tail)]))
        )
    for head, relation in true_tail:
        true_tail[(head, relation)] = np.array(
            list(set(true_tail[(head, relation)]))
        )

    return true_head, true_tail

# def test_get_true_head_and_tail():
#     test_triples = [
#         [0, 1, 2],
#         [0, 1, 3],
#         [3, 1, 2],
#     ]
#     true_head, true_tail = get_true_head_and_tail(test_triples)
#     print(f'true_head: {true_head}, true_tail: {true_tail}')

# def get_weight(triples,idx,count):
#     import torch
#     # 1. 正样本
#     head, relation, tail = triples[idx]
#
#     # 2. 权重（直接用传入的 count）
#     subsampling_weight = count[(head, relation)] + count[(tail, -relation - 1)]
#     subsampling_weight = torch.sqrt(1 / torch.tensor([subsampling_weight]))
#
#     return subsampling_weight

def get_negative_sample(triples, idx, negative_size, nentity, mode, true_head, true_tail):
    #生成所有mode对应的负样本取值
    import torch,numpy as np

    positive_sample = triples[idx]
    head, relation, tail = positive_sample

    negative_sample_list = []
    negative_sample_size = 0

    while negative_sample_size < negative_size:
        negative_sample = np.random.randint(nentity, size=negative_size * 2)

        if mode == "head-batch":
            mask = np.isin(negative_sample, true_head[(relation, tail)], invert=True)
        elif mode == "tail-batch":
            mask = np.isin(negative_sample, true_tail[(head, relation)], invert=True)
        else:
            raise ValueError("not support")

        negative_sample = negative_sample[mask]

        negative_sample_list.append(negative_sample)
        negative_sample_size += negative_sample.size

    negative_sample = np.concatenate(negative_sample_list)[:negative_size]

    negative_sample = torch.LongTensor(negative_sample)

    return negative_sample, mode

# def test_get_negetive_sample():
#     # 模拟数据集
#     test_triples = [[0, 1, 2], [0, 1, 3], [4, 5, 6]]
#     nentity = 10
#     mode = "head-batch"
#     negative_size = 8
#
#     count = count_frequency(test_triples)
#     true_head, true_tail = get_true_head_and_tail(test_triples)
#     a,b = get_negetive_sample(
#         test_triples,
#         1,
#         negative_size,
#         nentity,
#         mode,
#         true_head,
#         true_tail
#     )
#     print(f'a:{a}, b:{b}')

class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data

# def test_bidirectional_iterator():
#     import torch
#     # 造两个假数据集
#     data_head = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # head 批次数据
#     data_tail = torch.tensor([[99, 99], [88, 88], [77, 77]])     # tail 批次数据
#
#     # 做成 DataLoader
#     loader_head = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_head), batch_size=1)
#     loader_tail = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_tail), batch_size=1)
#
#     # 用我们的迭代器交替读取
#     multi_iterator = BidirectionalOneShotIterator(loader_head, loader_tail)
#     for i in range(10):
#         print(next(multi_iterator))

class OneShotIterator(object):
    def __init__(self,data):
        self.iterator = self.one_shot_iterator(data)
    def __next__(self):
        data = next(self.iterator)
        return data
    @staticmethod
    def one_shot_iterator(data):
        while True:
            for d in data:
                yield d

# def test_OneShotIterator():
#     import torch
#     data = [[1,2,3],[4,5,6],[7,8,9]]
#     iterator = OneShotIterator(data)
#     for i in range(10):
#         print(next(iterator))

def reverse_dict(file_path,delimiter="\t"):
    k2v = {}
    with open(file_path,'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(delimiter)
            if len(parts) != 2:
                continue

            k,v = parts[0],parts[1]

            k2v[k] = v
        v2k = {v:k for k,v in k2v.items()}
    return k2v,v2k

# def test_reverse_dict():
#     import random
#     a = ["北京","上海","中国"]
#     b = ["首都","位于"]
#     with open('entity.dict','w') as f:
#         for i in range(10):
#             f.write("0\t北京\n")
#             f.write("1\t上海\n")
#             f.write("2\t中国\n")
#     k2v,v2k = reverse_dict('entity.dict')
#     print(f'k2v:{k2v},v2k:{v2k}')

# def build_parser(name,type,default,help,choices):
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         name,
#         type=type,
#         default=default,
#         help=help,
#         choices=choices,
#         required=False,
#     )
#     return parser.parse_args()
# print(build_parser('--a',type=str,default='b',help='help',choices=['a','b']))

def fill_respective_empty_place(data,pad_val=-1):
    import torch
    result = []
    for i in range(len(data[0])):
        get = [sample[i] for sample in data]
        get_tensor = [torch.tensor(x) for x in get]
        max_len = max((ids.numel() for ids in get_tensor), default=0)
        padded = []
        for t in get_tensor:
            if t.dim() == 0:  # 标量，不填充
                padded.append(t)
            else:  # 列表，填充
                t_padded = torch.nn.functional.pad(t, (0, max_len - t.numel()), value=pad_val)
                padded.append(t_padded)
        result.append(torch.stack(padded))
    return result

# def test_fill_respective_empty_place():
#     fake_data = [
#             # (idx, cluster_id, neighbor_clusters_ids, parent_ids)
#             (0, 100, [1, 2, 3],      [10]),
#             (1, 200, [4, 5],         [11, 12, 13]),
#             (2, 300, [6],            [14, 15]),
#         ]
#     print(fill_respective_empty_place(fake_data))
