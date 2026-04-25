import numpy as np,torch,os,logging,argparse

def count_frequency(triples, start=4):
    #head, relation或者tail, -relation - 1，出现一次count++
    count = {}
    for head, relation, tail in triples:
        if (head, relation) not in count:
            count[(head, relation)] = start
        else:
            count[(head, relation)] += 1

        if (tail, -relation - 1) not in count:
            count[(tail, -relation - 1)] = start
        else:
            count[(tail, -relation - 1)] += 1
    return count

# triples = [
#     (0,1,2),
#     (0,1,3),
#     (1,2,3)
# ]
# print(count_frequency(triples))

def get_true_head_and_tail(triples):
    #已知head, relation得tail，已知relation, tail得head
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

# triples = [
#     (0,1,2),
#     (0,1,3),
#     (1,2,3)
# ]
# print(get_true_head_and_tail(triples))

def get_subsampling_weight(triples,idx,count):
    #count_frequency越大subsampling_weight越小
    #需要count_frequency
    positive_sample = triples[idx]

    head, relation, tail = positive_sample

    subsampling_weight = (
            count[(head, relation)] + count[(tail, -relation - 1)]
    )
    subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
    return subsampling_weight

# triples = [
#     (0,1,2),
#     (0,1,3),
#     (1,2,3)
# ]
# count = count_frequency(triples)
# print(get_subsampling_weight(triples,0,count))
# print(get_subsampling_weight(triples,1,count))
# print(get_subsampling_weight(triples,2,count))


def get_negative_samples(triples,idx,self_negative_sample_size,self_nentity,self_mode,self_true_head,self_true_tail):
    #生成负样本
    #triples多个三元组
    #idx取某个三元组
    #self_negative_sample_size要多大的负样本
    #self_nentity负样本取值范围0~self_nentity
    #self_mode要头的还是尾巴的负样本
    #self_true_head,self_true_tail用来做比较，要调用get_true_head_and_tail
    positive_sample = triples[idx]

    head, relation, tail = positive_sample

    negative_sample_list = []
    negative_sample_size = 0

    while negative_sample_size < self_negative_sample_size:
        negative_sample = np.random.randint(
            self_nentity, size=self_negative_sample_size * 2
        )
        if self_mode == "head-batch":
            mask = np.isin(
                negative_sample,
                self_true_head[(relation, tail)],
                assume_unique=True,
                invert=True,
            )
        elif self_mode == "tail-batch":
            mask = np.isin(
                negative_sample,
                self_true_tail[(head, relation)],
                assume_unique=True,
                invert=True,
            )
        else:
            raise ValueError("Training batch mode %s not supported" % self_mode)
        negative_sample = negative_sample[mask]
        negative_sample_list.append(negative_sample)
        negative_sample_size += negative_sample.size

    negative_sample = np.concatenate(negative_sample_list)[
        : self_negative_sample_size
    ]

    negative_sample = torch.LongTensor(negative_sample)
    return negative_sample

# triples = [
#     (0,1,2),
#     (0,1,3),
#     (1,2,3)
# ]
# self_true_head, self_true_tail = get_true_head_and_tail(triples)
# self_mode = "head-batch"
# self_negative_sample_size = 2
# self_nentity = 20
# idx = 1
# a = get_negative_samples(triples,idx,self_negative_sample_size,self_nentity,self_mode,self_true_head,self_true_tail)
# print(a)

def collate_fn(data):
    #把data对应的部分合一起
    positive_sample = torch.stack([_[0] for _ in data], dim=0)
    negative_sample = torch.stack([_[1] for _ in data], dim=0)
    subsample_weight = torch.cat([_[2] for _ in data], dim=0)

    mode = data[0][3]
    return (
        positive_sample,
        negative_sample,
        subsample_weight,
        mode,
    )

# data = [
#     # 第1条样本
#     (
#         torch.LongTensor([1, 0, 2]),  # 0: 正样本
#         torch.LongTensor([3, 4]),     # 1: 负样本
#         torch.Tensor([0.5]),          # 2: 权重
#         "tail-batch"                  # 3: mode
#     ),
#     # 第2条样本
#     (
#         torch.LongTensor([2, 0, 3]),  # 0: 正样本
#         torch.LongTensor([1, 5]),     # 1: 负样本
#         torch.Tensor([0.8]),          # 2: 权重
#         "tail-batch"                  # 3: mode
#     ),
# ]
# print(collate_fn(data))

# def getitem(triples, idx, mode,triple_set,nentity,rerank,alpha,k_hop_neighbors_head,k_hop_neighbors_tail):
#     head, relation, tail = triples[idx]
#
#     if mode == "head-batch":
#         tmp = [
#             (0, rand_head)
#             if (rand_head, relation, tail) not in triple_set
#             else (-1, rand_head)
#             for rand_head in range(nentity)
#         ]
#         tmp[head] = (0, head)
#         tmp[tail] = (-1, tail)
#     elif mode == "tail-batch":
#         tmp = [
#             (0, rand_tail)
#             if (head, relation, rand_tail) not in triple_set
#             else (-1, rand_tail)
#             for rand_tail in range(nentity)
#         ]
#         tmp[tail] = (0, tail)
#         tmp[head] = (-1, head)
#     else:
#         raise ValueError("negative batch mode %s not supported" % mode)
#     print(tmp)
#     tmp = torch.LongTensor(tmp)
#     filter_bias = tmp[:, 0].float()
#
#     if rerank:
#         if mode == "tail-batch":
#             k_hop_neighbors_head = torch.LongTensor(k_hop_neighbors_head)
#             filter_bias[k_hop_neighbors_head] += alpha
#         elif mode == "head-batch":
#             k_hop_neighbors_tail = torch.LongTensor(k_hop_neighbors_tail)
#             filter_bias[k_hop_neighbors_tail] += alpha
#
#     negative_sample = tmp[:, 1]
#
#     positive_sample = torch.LongTensor((head, relation, tail))
#
#     return (
#         positive_sample,
#         negative_sample,
#         filter_bias,
#         mode,
#     )
#
# triples = [
#     (0, 0, 1),   # 正三元组 0
#     (1, 0, 2)
# ]
# idx = 0                 # 取第一个三元组
# mode = "tail-batch"     # 测试替换尾实体
# triple_set = set((h, r, t) for h, r, t in triples)
# nentity = 3            # 假设有 3 个实体：0,1,2
# rerank = True
# alpha = 0.5
# k_hop_neighbors_head = [1]  # 头实体 0 的邻居
# k_hop_neighbors_tail = [2]
# print(getitem(triples, idx, mode,triple_set,nentity,rerank,alpha,k_hop_neighbors_head,k_hop_neighbors_tail))


class BidirectionalOneShotIterator(object):
    #你一个我一个地取
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

# dataloader_head = (101, 102, 103)   # head 数据
# dataloader_tail = [201, 202, 203]   # tail 数据
# iter_loader = BidirectionalOneShotIterator(dataloader_head, dataloader_tail)
# print(iter_loader)
# # 连续取 6 次数据
# for i in range(6):
#     res = next(iter_loader)
#     print(f"第{i+1}次取出: {res}")

class OneShotIterator(object):
    #一个一个地取
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)

    def __next__(self):
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data

# a = OneShotIterator(dataloader_head)
# for i in range(len(dataloader_head)):
#     print(next(a))

def make_entities_and_relations():
    #给read_data_from_args造例子的
    class Args:
        def __init__(self):
            self.data_path = "./test_data"  # 数据文件夹

    args = Args()

    os.makedirs(args.data_path, exist_ok=True)

    with open(os.path.join(args.data_path, "entities.dict"), "w") as f:
        f.write("0\t小明\n")
        f.write("1\t小红\n")
        f.write("2\t学校\n")
        f.write("3\t北京\n")

    with open(os.path.join(args.data_path, "relations.dict"), "w") as f:
        f.write("0\t喜欢\n")
        f.write("1\t在...里\n")
        f.write("2\t朋友\n")

def read_data_from_args(args):
    #从路径的文件里读出实体：ID，关系：ID等，文件里的格式必须是xxx\txxx
    with open(os.path.join(args, "entities.dict")) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split("\t")
            entity2id[entity] = int(eid)
        id2entity = {v: k for k, v in entity2id.items()}

    with open(os.path.join(args, "relations.dict")) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split("\t")
            relation2id[relation] = int(rid)
        id2relation = {v: k for k, v in relation2id.items()}

    nentity = len(entity2id)
    nrelation = len(relation2id)
    logging.basicConfig(level=logging.INFO)
    logging.info("Data Path: %s" % args)
    logging.info("#entity: %d" % nentity)
    logging.info("#relation: %d" % nrelation)

    return {
        "entity2id": entity2id,
        "id2entity": id2entity,
        "relation2id": relation2id,
        "id2relation": id2relation,
    }

# make_entities_and_relations()
# args = './test_data'
# print(read_data_from_args(args))
# print([read_data_from_args(args)['id2entity'][i]for i in range(4)])

def build_data_args(parser):
    #加参数的
    parser.add_argument(
        "--data_path", type=str, default="data", help="Path to the dataset"
    )
    parser.add_argument(
        "--process_path",
        type=str,
        default="processed_data",
        help="Path to the entity hierarchy",
    )
    parser.add_argument("--dataset", type=str, default="FB15K-237", help="Dataset name")
    parser.add_argument(
        "--hierarchy_type",
        type=str,
        default="seed",
        choices=["seed", "llm"],
        help="Type of hierarchy to use",
    )
# parser = argparse.ArgumentParser()
# build_data_args(parser)
# args = parser.parse_args()
# print(args.dataset)

# def collate_fn1(data):
#     #把对于位置的都合成一个tensor，形状不一样的后面填充-1
#     cluster_id = torch.tensor([sample[1] for sample in data], dtype=torch.long)
#     id = torch.tensor([sample[0] for sample in data], dtype=torch.long)
#
#     # neighbor clusters padding
#     neighbor_clusters_ids = [torch.LongTensor(sample[2]) for sample in data]
#     max_len_neighbor = max(
#         (ids.numel() for ids in neighbor_clusters_ids), default=0
#     )
#     padded_neighbor_clusters_ids = (
#         torch.stack(
#             [
#                 torch.nn.functional.pad(
#                     ids, (0, max_len_neighbor - ids.numel()), value=-1
#                 )
#                 for ids in neighbor_clusters_ids
#             ]
#         )
#         if len(neighbor_clusters_ids) > 0
#         else torch.empty((0, max_len_neighbor), dtype=torch.long)
#     )
#
#     # parent ids padding
#     parent_ids = [torch.LongTensor(sample[3]) for sample in data]
#     max_len_parent = max((ids.numel() for ids in parent_ids), default=0)
#     padded_parent_ids = (
#         torch.stack(
#             [
#                 torch.nn.functional.pad(
#                     ids, (0, max_len_parent - ids.numel()), value=-1
#                 )
#                 for ids in parent_ids
#             ]
#         )
#         if len(parent_ids) > 0
#         else torch.empty((0, max_len_parent), dtype=torch.long)
#     )
#
#     return id, cluster_id, padded_neighbor_clusters_ids, padded_parent_ids
#
# # data_input = [
# #     (0, 10, [11, 12], [20]),
# #     (1, 11, [13],    [21, 22]),
# #     (2, 12, [14,15,16], [23])
# # ]
# # print(collate_fn1(data_input))