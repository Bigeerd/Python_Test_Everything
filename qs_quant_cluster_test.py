def keysvalues():
    entity_info = {
        "苹果": {"描述": "1", "向量": "2", },
        "汽车": {"描述": "1", "向量": "2", },
        "太阳": {"描述": "1", "向量": "2", },

    }
    print(entity_info.keys())
    print(entity_info.values())
    entities = entity_info.keys()
    print([entity_info[entity]['描述'] for entity in entities])
    discriptions = [entity_info[entity]['向量'] for entity in entities]
    for i in range(len(discriptions)):
        print(discriptions[i])

def fileop():
    import os
    print(os.path.exists('./.venv'))
    arg1 = 'cluster'
    arg2 = 'llm_refine'
    arg3 = 'check.py'
    print(os.path.exists(f'./{arg1}/{arg2}/{arg3}'))
    print(os.path.exists('./'))
    with open(f'./{arg1}/{arg2}/README.md','r') as f:
        print(f.read())

def createjson():
    import json
    data = {
        "name": "苹果",
        "description": "苹果是一种水果",
        "vector": [0.1, 0.2, 0.3]
    }
    with open('test.json', 'w',encoding='utf-8') as f:
        json.dump(data, f)
    with open('test.json', 'r',encoding='utf-8') as f1:
        data = json.load(f1)
    print(data)

def ifelse():
    bool = True
    a = 1 if bool else 0
    print(a)

def dotdotdot():
    class A(object):
        def __init__(self):
            self.a = 1
    class B(object):
        def __init__(self):
            self.b = A()
    c = B()
    d = (
        c
        .b
        .a
    )
    print(d)

def concatenate():
    import numpy as np
    np1 = np.array([[1,1], [2,2], [3,3]])
    np2 = np.array([[4,4], [5,5], [6,6]])
    np3 = np.concatenate((np1,np2))
    print(np3)
    print(np.concatenate([np1,np2]))

def tolist():
    import numpy as np
    np1 = np.array([1,2,3])
    print(type(np1))
    print(np1)
    print(type(np1.tolist()))
    print(np1.tolist())

def multithread():
    from concurrent.futures import ThreadPoolExecutor
    import time
    def print1(a):
        time.sleep(0.5)
        print(f'{a}')

    start = time.time()
    print1(1)
    print1(2)
    print1(3)
    print(time.time() - start)

    start = time.time()
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.submit(print1,1)
        executor.submit(print1,2)
        executor.submit(print1, 3)
    print(time.time() - start)

def multithread2():
    from concurrent.futures import ThreadPoolExecutor
    def f1(i):
        return i

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(f1,i)for i in range(10)]
        results = [r.result() for r in futures]
        print(results)

def tqdm():
    import tqdm,time

    for i in tqdm.tqdm(
            range(100),#进度条任务
            total=100,#进度条的范围
            desc="generating",#进度条前面的说明文字
            leave=False,#跑完进度条消失
            ncols=100,#进度条宽度
    ):
        time.sleep(0.01)


def ascompleted():
    from concurrent.futures import as_completed, ThreadPoolExecutor
    import time

    def f1(x):
        if x > 1:
            time.sleep(2)
        elif x == 1:
            time.sleep(1)
        else:
            time.sleep(0.5)
        return x

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(f1,i)for i in range(2,-1,-1)
        ]
        for f in futures:
            print(f.result())
        for f in as_completed(futures):
            print(f.result())

def testfutureandtqdm():
    from concurrent.futures import as_completed,ThreadPoolExecutor
    from tqdm import tqdm
    import time
    def f1(x):
        time.sleep(0.5)
        return x,x+1
    with ThreadPoolExecutor(max_workers=10) as executor:
        len1 = 10
        futures = [executor.submit(f1,i)for i in range(len1)]
        print(futures)
        for future in tqdm(as_completed(futures),total=len1):
            a,b = future.result()
            print(a,b)

def idx():
    entities = ["实体A", "实体B", "实体C"]
    futures = ["futureA", "futureB", "futureC"]
    entity = 'futureA'
    entity_info = {}
    print(futures.index(entity))
    print(entities[futures.index(entity)])
    entity_info[entities[futures.index(entity)]] = entity
    print(entity_info)

def dot2f():
    dis = 0.123456
    i = 4
    print(dis)
    print(f'{dis:.{i}f}')

def Aggcluster():
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    X = np.array([
        [1, 0],
        [1, 0.1],
        [0, 1],
        [0.1, 1],
        [0, 0.9]
    ])

    clustering = AgglomerativeClustering(
        metric='cosine',
        linkage='average',
        distance_threshold=0.1,
        n_clusters=None,
    )

    re1 = clustering.fit_predict(X)
    print(f'clustering.fit_predict : {re1}')
    clustering.fit(X)
    print(f'clustering.labels_ : {clustering.labels_}')
    print(f'clustering.n_clusters_ : {clustering.n_clusters_}')
    for i in range(clustering.n_clusters_):
        print(f"type : {type(np.where(clustering.labels_ == i))}")
        print(f"np.where(clustering.labels_ == i) : {np.where(clustering.labels_ == i)}")
        print(f"np.where(clustering.labels_ == i)[0] : {np.where(clustering.labels_ == i)[0]}")

    import pickle
    with open("model.pkl", "wb") as f:
        pickle.dump(clustering, f)
    with open("model.pkl", "rb") as f:
        loaded_clustering = pickle.load(f)
    print(f'loaded_clustering.labels_ : {loaded_clustering.labels_}')

def wrb():
    with open('123_test','w',encoding='utf-8') as f:
        for i in range(100):
            f.write(f'{i}')
            f.write('\n')
    with open('123_test','r',encoding='utf-8') as f:
        lines = f.readlines()
        print(lines)

def lam():

    print([(lambda x : x * x)(i) for i in range(10)] )

    fuc = lambda x : x * x
    print([fuc(i)for i in range(10)])

def addcluster():
    cluster = {}
    for i in range(10):
        cluster[f'cluster_{i+1}'] = i + 1
    print(cluster)

def addsth():
    cluster = []
    for i in range(10):
        cluster.append(f'cluster_{i+1}')
    print(cluster)

def hei():
    cluster = {f'cluster_{i+1}' : i+1 for i in range(10)}
    print(cluster)
    print(cluster.pop('cluster_1' if cluster['cluster_1'] != 1 else 'cluster_2')  )
    print(cluster)

def testmerge():
    hie = {
        "Leaf_0": "实体A",
        "Leaf_1": "实体B",
        "Leaf_2": "实体C",
    }
    n_leaves = len(hie)

    left = 0
    right = 1
    cluster_id = "cluster_100"

    hie[cluster_id] = {
        f'cluster_{left}' : hie.pop(
            f'Leaf_{left}' if left < n_leaves else f'cluster_{left}'
        ),
        f'cluster_{right}' : hie.pop(
            f'Leaf_{right}' if right < n_leaves else f'cluster_{right}'
        ),
    }
    print(hie)

def merge_leaves():
    hierarchy = {
        "leaf_0": "苹果",
        "leaf_1": "香蕉",
        "leaf_2": "橙子",
    }
    n_leaves = len(hierarchy)
    next_cluster_id = n_leaves

    children = [
        (0, 1),# 第一次：合并 0 和 1
        (2, 3),# 第二次：合并 2 和 3（3是上一轮生成的簇）
    ]

    for i,(left,right) in enumerate(children):

        hierarchy[f'cluster_{next_cluster_id}'] = {
            f'cluster_{left}' : hierarchy.pop(
                f'leaf_{left}' if left < n_leaves else f'cluster_{left}'
            ),
            f'cluster_{right}' : hierarchy.pop(
                f'leaf_{right}' if right < n_leaves else f'cluster_{right}'
            ),
        }
        next_cluster_id += 1
    print(hierarchy[f"cluster_{next_cluster_id - 1}"])
    print(hierarchy)

def build():
    cluster = {
        "Cluster_1": ["苹果", "香蕉"],
        "Cluster_2": ["橙子", "葡萄"],
        "Cluster_3": ["西瓜"]
    }

    clusters_ = {int(i+1) : entity for i,entity in enumerate(cluster.values())}
    print(clusters_)
    entity2clusterid = {}
    for i,cluster in enumerate(clusters_.values()):
        for entity in cluster:
            entity2clusterid[entity] = i + 1
    print(entity2clusterid)

def reverse_dict():
    dict = {
        'a1': 'b1',
        'a2': 'b2',
        'a3': 'b3',
        'a4': 'b4',
        'a5': 'b5',
        'a6': 'b6',
        'a7': 'b7',
        'a8': 'b8',
        'a9': 'b9',
        'a10': 'b10',
    }
    reverse1 = {}
    for entity in dict.keys():
        reverse1[dict[entity]] = entity
    print(reverse1)

def defauledict():
    from collections import defaultdict
    dict = defaultdict(float)
    print(dict['apple'])

def logging():
    import logging
    logging.basicConfig(level=logging.INFO)
    a = 0.123456789
    logging.info(f"{a:.3f}")

def npdotzeros():
    import numpy as np
    print(np.zeros((2,2,2),dtype=int))
    print(np.zeros(1))

def silhouette():
    import numpy as np
    import sklearn
    embeddings = [
        [1,1],
        [1,1],
        [10,99]
    ]
    embeddings = np.array(embeddings)
    labels = [0,0,1]
    score = sklearn.metrics.silhouette_score(embeddings, labels)#分的越好打分越高
    print(score)

def linspace():
    import numpy as np
    print(np.linspace(1,2,30))

def maxthread():
    import concurrent.futures
    def f1(a,b,c,d,e,f,g):
        print(a,b,c,d,e,f,g)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(f1, 1, 2, 3, 4, 5, 6, 7),
            executor.submit(f1, 7, 8, 9, 10, 11, 12, 13),
            executor.submit(f1, 10, 11, 12, 13, 14, 15, 16),
            executor.submit(f1, 17, 18, 19, 20, 21, 22, 23),
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

def write():
    import time
    a = []
    with open(f'bibilabu.txt', 'w') as f:
        for i in range(100):
            f.write(f'  {time.time() + i}  \n')
    with open(f'bibilabu.txt', 'r') as f:
        for i in range(100):
            a.append(f.readline().strip())
    with open(f'bibilabu.txt', 'r') as f:
        b = f.readlines()
    c = []
    for line in b:
        c.append(line.strip())
    if a == c:
        print('ok!!!')

def rmtab():
    a = 'asffasdfasdf\n'
    print([a])
    print([a.strip()])

def jsonload():
    data = {
        'a1我' : 'b1',
        'a2' : 'b2',
        'a3' : 'b3我',
        'a4' : 'b4',
        'a5' : 'b5',
        'a6' : 'b6',
        'a阿斯蒂芬7' : 'b7',
        'a8' : 'b8',
        'a9' : 'b9',
        'a10' : 'b10',
    }
    import json
    with open('bibilabu.json', 'w') as f:
        json.dump(data,f,indent=4,ensure_ascii=False)
    with open('bibilabu.json', 'r') as f:
        a = json.load(f)
    print(a)

def repr1():
    a = '   apple   banana\n'
    print(repr(a))
    b,c = a.strip().split('   ')
    print(b,c)
    print(a.strip().split('   ')[0])
    print(a.strip().split('   ')[1])

def fuzhi():
    (
        a,
        b,
        c,
        __,
        _,
    ) = [1,2,3,4,5]
    print(a,b,c,__,_)

def isininin():
    if str(1) in str(1234):
        print(1)
    if 1 in [1,2,3,4]:
        print(1)

def stradd():
    str1 = 'a ,b ,c ,d ,e ,f ,g ,h'
    a = ','
    for data in str1.split(' ' + a):
        print(data)

def dedict():
    from collections import defaultdict
    dict = defaultdict(list)
    print(dict['apple'])
    print(dict)

    entity_info = {
        "苹果": {"text_label": "水果"},
        "香蕉": {"text_label": "水果"},
        "小狗": {"text_label": "动物"},
        "小猫": {"text_label": "动物"},
    }

    for entity in entity_info.keys():
        dict[entity_info[entity]['text_label']].append(entity)
    print(dict)

def testtqdm():
    from tqdm import tqdm
    import time
    len1 = 10
    for i in tqdm(range(len1),desc='hahha',ncols=100):
        time.sleep(0.1)

def addargg():
    def addarg():
        import argparse
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--a",# 参数名（前面加 --）
            type=int,# 参数类型：int/float/str
            default=1,# 默认值（不传就用这个）
            choices=[1,2,3,4,5],
            required=False,# 是否必填（True 必须传）
            help="wocao"# 说明文字
        )
        parser.add_argument(
            '--c',
            type=str,
            required=False,
            default='bee',
            choices=['bee','bee2','bee3','bee4'],
            help="bee"
        )
        args = parser.parse_args()
        args.b = args.a
        return args

    arg = addarg()
    print(arg.a)
    print(arg.b)
    print(arg.c)

def loggerr():
    import logging,time
    file_name = time.strftime('%Y%m%d_%H%M%S') + ' log.txt'
    logging.basicConfig(
        filename = file_name,
        level=logging.INFO,
        format = '%(asctime)s - %(levelname)s - %(message)s',
    )
    logging.info('test')

def percent():
    data = {
        "name": "小明",  # x 可以是 name
        "age": 18,  # x 可以是 age
        "city": "北京"  # x 可以是 city
    }
    print('%(name)s - %(age)s - %(city)s' % data)


def ret():
    def f1():
        return 'a','b','c','d','e','f'
    a,_,_,_,_,f = f1()
    print(a,f)
    print(_)

def labvel():
    l2e = {
        'a1': {'b1' : 'c1'},
        'a2': {'b2' : 'c2'},
        'a3': {'b3' : 'c3'},
    }
    a = {
        list(entity2.values())[0] : entity1 for entity1,entity2 in l2e.items()
    }
    print(a)

def toli():
    import torch,numpy as np
    a = torch.Tensor([1,2,3,4,5])
    b = np.array([1,2,3,4,5])
    print(type(a),type(b))
    print(a.tolist())
    print(b.tolist())

def npsaveload():
    import numpy as np,time
    arr = [[1,2,3],[4,5,6],[7,8,9]]
    arr = np.array(arr)
    np.save(
        file = time.strftime('%Y%m%d') + '_save_np_arr.npy',
        arr = arr,
        allow_pickle = True,
    )
    load_arr = np.load(
        file = time.strftime('%Y%m%d') + '_save_np_arr.npy',
    )
    print(load_arr)
npsaveload()