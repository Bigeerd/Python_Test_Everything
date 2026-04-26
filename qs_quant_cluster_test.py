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
    print(re1)
    clustering.fit(X)
    print(clustering.labels_)

    import pickle
    with open("model.pkl", "wb") as f:
        pickle.dump(clustering, f)
    with open("model.pkl", "rb") as f:
        loaded_clustering = pickle.load(f)
    print(loaded_clustering.labels_)

def wrb():
    with open('123_test','w',encoding='utf-8') as f:
        for i in range(100):
            f.write(f'{i}')
            f.write('\n')
    with open('123_test','r',encoding='utf-8') as f:
        lines = f.readlines()
        print(lines)
