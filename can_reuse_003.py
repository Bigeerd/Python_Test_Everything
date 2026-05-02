# DEPENDENCY: 需要本地运行 Ollama，且已 pull qwen2:0.5b 和 mxbai-embed-large
# 启动命令: ollama serve
# 下载模型: ollama pull qwen2:0.5b && ollama pull mxbai-embed-large

entities = ["Apple", "Orange", "strawberry", "Banana", "Computer", "Phone", "Airplane"]
test_tree = {'Cluster_12': {'Cluster_6': 'Airplane', 'Cluster_11': {'Cluster_9': {'Cluster_2': 'strawberry', 'Cluster_3': 'Banana'}, 'Cluster_10': {'Cluster_7': {'Cluster_0': 'Apple', 'Cluster_1': 'Orange'}, 'Cluster_8': {'Cluster_4': 'Computer', 'Cluster_5': 'Phone'}}}}}

from openai import OpenAI
client = OpenAI(api_key="ollama_is_free", base_url="http://localhost:11434/v1")

def get_AI_response(model, prompt, seed=44):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0,
            seed=seed,
        ).choices[0].message.content
        return response
    except Exception as e:
        print(e)

# def test_get_AI_response():
#     from openai import OpenAI
#     # api_key = "ollama_is_free"
#     # base_url = "http://localhost:11434/v1"
#     # client = OpenAI(base_url=base_url, api_key=api_key)
#     embedding_model = "bge-m3"
#     chat_model = "qwen2:0.5b"
#     try:
#         resp = gpt_AI_response(chat_model, "你好")
#         print("✅ 本地 Ollama 调用成功！")
#         print("返回内容：", resp.choices[0].message.content)
#     except Exception as e:
#         print("❌ 报错：", e)

def generate_entity_description(entity, hint=None):
    #rely:gpt_AI_response
    if hint:
        prompt = f"Please provide a brief description of the entity '{entity}' in the following format:\n\n{entity} is a [description].\n\nFor example:\napple is a round fruit with red, green, or yellow skin and crisp, juicy flesh.\n\nHINT:{hint}\n\nNow, describe {entity}:"
    else:
        prompt = f"Please provide a brief description of the entity '{entity}' in the following format:\n\n{entity} is a [description].\n\nFor example:\nBill Gates is a technology magnate, philanthropist, and co-founder of Microsoft Corporation, known for his significant contributions to the personal computing industry.\n\nNow, describe {entity}:"
    response = get_AI_response(model="qwen2:0.5b", prompt=prompt)
    description = response.choices[0].message.content.strip()
    return description

# def test_generate_entity_description():
#     dis = generate_entity_description(
#         entity="apple",
#         hint="trump?",
#     )
#     print(dis)

def get_embedding(text,model="mxbai-embed-large"):
    #"nomic-embed-text"
    #"bge-m3"
    #ollama pull mxbai-embed-large --insecure
    return client.embeddings.create(
        input=text, model = model
    ).data[0].embedding

def mix_entity_and_description_embedding(entity, description):
    import numpy as np
    emb1 = get_embedding(entity)
    emb2 = get_embedding(description)
    return np.concatenate([emb1, emb2])

# def test_generate_entity_embedding():
#     print("正在测试嵌入生成...")
#     entity = "Apple"
#     description = "Apple is a technology company that designs smartphones and computers."
#     final_embedding = generate_entity_embedding(entity, description)
#     print("✅ 实体:", entity)
#     print("✅ 描述:", description)
#     print("✅ 最终向量长度:", len(final_embedding))
#     print("✅ 向量前5个值:", final_embedding[:5])

def clustering(embeddings,distance_threshold=0.4,metric="cosine",linkage="average"):
    import sklearn
    clustering = sklearn.cluster.AgglomerativeClustering(
        metric=metric,
        linkage=linkage,
        distance_threshold=distance_threshold,
        n_clusters=None,
    )
    clustering.fit(embeddings)
    return clustering

# def save_clustering(clustering,output_file):
#     try:
#         with open(output_file,"wb") as f:
#             pickle.dump(clustering,f)
#         print("Saved clustering!!!")
#     except Exception as e:
#         print(e)
# def load_clustering(output_file):
#     try:
#         with open(output_file,"rb") as f:
#             clustering = pickle.load(f)
#         print(clustering)
#         return clustering
#     except Exception as e:
#         print(e)
#
# def test_clustering():
#     import numpy as np
#     embeddings = np.array([
#             [1, 2, 3],
#             [1, 1, 3],
#             [10, 11, 12],
#             [10, 12, 13],
#             [100, 200, 300]
#         ])
#     cluster = clustering(embeddings,0.001)
#     save_clustering(clustering,"clustering.pkl")
#     clu = load_clustering('clustering.pkl')
#     print(clu.labels_)

def clustering2dict(entities,embeddings, distance_threshold=0.4):
    #entities=["Apple", "Banana", "Computer", "Phone", "Airplane"]
    #rely clustering()
    import numpy as np
    cluster = clustering(
        embeddings, distance_threshold
    )

    clusters = {}
    for i in range(cluster.n_clusters_):
        cluster_indices = np.where(cluster.labels_ == i)[0]
        cluster_entities = [entities[idx] for idx in cluster_indices]
        clusters[f"Cluster_{i + 1}"] = cluster_entities

    return clusters

def build_tree(children, entities):
    n_leaves = len(entities)
    hierarchy = {f"Leaf_{i}": entities[i] for i in range(n_leaves)}
    next_cluster_id = n_leaves
    for i, (left, right) in enumerate(children):
        cluster_id = f"Cluster_{next_cluster_id}"
        hierarchy[cluster_id] = {
            f"Cluster_{left}": hierarchy.pop(
                f"Leaf_{left}" if left < n_leaves else f"Cluster_{left}"
            ),
            f"Cluster_{right}": hierarchy.pop(
                f"Leaf_{right}" if right < n_leaves else f"Cluster_{right}"
            ),
        }
        next_cluster_id += 1
    return hierarchy

def tree_label(tree, entity2clusterid, clusterid2count):
    #[x, y]x=entity2clusterid,y=第几次出现了
    for key, value in tree.items():
        if isinstance(value, dict):  # If the value is another dictionary, recurse into it
            tree_label(value, entity2clusterid, clusterid2count)
        else:  # If the value is not a dictionary, then it's a leaf node
            cluster_id = entity2clusterid[value]
            tree[key] = [cluster_id, clusterid2count[cluster_id]]
            clusterid2count[entity2clusterid[value]] += 1
    return tree

def tree_clean_1(tree, clusters_):
    #清理同级节点
    keys_to_delete = []
    items_to_update = {}

    for key, value in list(tree.items()):
        if isinstance(value, dict):
            tree_clean_1(value, clusters_)
        else:
            if value[1] > 0:
                keys_to_delete.append(key)
            else:
                items_to_update[key] = clusters_[value[0]]

    # Now, delete keys marked for deletion
    for key in keys_to_delete:
        del tree[key]

    # Update the dictionary with new values
    for key, new_value in items_to_update.items():
        tree[key] = new_value

    return tree

def tree_clean_2(tree):
    for key in list(tree.keys()):
        value = tree[key]
        if isinstance(value, dict):
            if value:
                result = tree_clean_2(value)
                if len(result) == 1 and isinstance(list(result.values())[0], list):
                    tree[key] = list(result.values())[0]
                else:
                    tree[key] = result
            else:
                del tree[key]
    return tree

def tree_clean_3(tree):
    new_tree = {}  # To accumulate refined results
    for key, value in list(tree.items()):
        if isinstance(value, dict):
            processed = tree_clean_3(value)  # Recursively process
            if processed:  # Only add non-empty results
                new_tree[key] = processed
        else:  # Keep non-dict items as they are
            new_tree[key] = value
    return new_tree

def tree_clean_4(tree):
    new_tree = {}
    for key, value in list(tree.items()):
        if isinstance(value, dict):
            processed = tree_clean_4(value)
            if processed:
                if isinstance(processed, dict) and len(processed) == 1:
                    new_tree[key] = list(processed.values())[0]
                else:
                    new_tree[key] = processed
        else:
            new_tree[key] = value
    return new_tree

# def test_tree():
#
#     from collections import defaultdict
#     entities = ["Apple", "Banana", "Computer", "Phone", "Airplane"]
#     entity_data = [
#         ("Apple", "A common fruit that grows on trees and is edible."),
#         ("Banana", "A long curved fruit with soft flesh and yellow skin."),
#         ("Computer", "An electronic device for processing data and running software."),
#         ("Phone", "A portable electronic device for communication and internet."),
#         ("Airplane", "A flying vehicle that transports people through the air.")
#     ]
#     embeddings = []
#     tmp = [entity_data[i][1] for i in range(len(entity_data))]
#     for i in range(len(entities)):
#         embeddings.append(mix_entity_and_description_embedding(entities[i], tmp[i]))
#     cluster_dict = clustering2dict(entities,embeddings,0.4)
#     #distance_threshold:0.4
#     #language:English
#     cluster = clustering(embeddings)
#
#     cluster_tree = build_tree(
#         children=cluster.children_,
#         n_leaves=len(entities),
#         entity_labels=entities,
#     )
#
#     entity2clusterid = {}
#     for cid,members in enumerate(cluster_dict.values()):
#         for member in members:
#             entity2clusterid[member] = cid
#
#     clusterid2count = defaultdict(int)
#
#     label = tree_label(cluster_tree, entity2clusterid, clusterid2count)
#
#     clusterid2entity = {}
#     for k,v in entity2clusterid.items():
#         if v not in clusterid2entity:
#             clusterid2entity[v] = []
#         clusterid2entity[v].append(k)
#     tree = tree_clean_1(label,clusterid2entity)
#     cleaned = tree_clean_2(tree)
#     removed = tree_clean_3(cleaned)
#     flattened = tree_clean_4(removed)
#     print(removed)
#     print(flattened)

def evaluate_threshold(entities, embeddings, threshold=0.36):
    import numpy as np,sklearn
    clusters = clustering2dict(entities, embeddings, threshold)
    num_clusters = len(clusters)

    if num_clusters == 1:
        return threshold, -1.0, None
    elif num_clusters == len(entities):
        return threshold, 0.0, None
    else:
        labels = np.zeros(len(entities), dtype=int)
        if type(clusters) is tuple:
            clusters = clusters[0]
        for i, cluster_entities in enumerate(clusters.values()):
            indices = [entities.index(entity) for entity in cluster_entities]
            labels[indices] = i
        score = sklearn.metrics.silhouette_score(embeddings, labels)
        return threshold, score, clusters

# def test_evaluate_threshold():
#     import ast
#     entities = ["Apple", "Orange", "strawberry", "Banana", "Computer", "Phone", "Airplane"]
#     entities_description = []
#     embeddings = []
#     for entity in entities:
#         entities_description.append(generate_entity_description(entity))
#     for i in range(len(entities)):
#         embeddings.append(mix_entity_and_description_embedding(entities[i], entities_description[i]))
#     print(evaluate_threshold(entities, embeddings))

def find_optimal_threshold(
    entities, embeddings, min_threshold=0.1, max_threshold=5, num_thresholds=100, num_threads=30
):
    import numpy as np,concurrent.futures
    thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_threads
    ) as executor:
        futures = [
            executor.submit(evaluate_threshold, entities, embeddings, threshold)
            for threshold in thresholds
        ]

        best_score = -1.0
        best_threshold = None
        best_clusters = None

        for future in concurrent.futures.as_completed(futures):
            threshold, score, clusters = future.result()
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_clusters = clusters

    if best_threshold is None:
        best_threshold = thresholds[-1]
        best_clusters = clustering2dict(
            entities, embeddings, best_threshold
        )

    return best_threshold, best_clusters

# def test_find_optimal_threshold():
#     entities = ["Apple", "Orange", "strawberry", "Banana", "Computer", "Phone", "Airplane"]
#     entities_description = []
#     embeddings = []
#     for entity in entities:
#         entities_description.append(generate_entity_description(entity))
#     for i in range(len(entities)):
#         embeddings.append(mix_entity_and_description_embedding(entities[i], entities_description[i]))
#     print(find_optimal_threshold(entities, embeddings))

def create_entity_info(entities):
    entity_info = {}
    for ent in entities:
        entity_info[ent] = {
            "text_label": ent,
            "original_description": None,
            "llm_description": None,
            "cluster": None,
            "parent_path": None,
            "nearest_clusters_lca": None
        }
    return entity_info

# def create_entity_emb(entities):
#     entity_embeddings = {}
#     for ent in entities:
#         entity_embeddings[ent] = None
#     return entity_embeddings
#
# def test_create_entity_info_and_emb():
#     entities = ["Apple", "Orange", "strawberry", "Banana", "Computer", "Phone", "Airplane"]
#     format = {
#                 "text_label": None,
#                 "original_description": None,
#                 "llm_description": None,
#             }
#     print(create_entity_emb(entities))

def find_leaves(tree, leaf_keys=None, leaf_values=None):
    if leaf_keys is None:
        leaf_keys = []
    if leaf_values is None:
        leaf_values = []
    for key, value in tree.items():
        if isinstance(value, dict):
            find_leaves(value, leaf_keys, leaf_values)
        else:
            leaf_keys.append(key)
            leaf_values.append(value)
    return leaf_keys, leaf_values

def build_hierarchy(entities,embeddings):
    cluster = clustering(embeddings)
    return build_tree(cluster.children_, entities)

# def test_find_leaves():
#     entities = ["Apple", "Orange", "strawberry", "Banana", "Computer", "Phone", "Airplane"]
#     entity_description = []
#     entity_embeddings = []
#     for entity in entities:
#         entity_description.append(generate_entity_description(entity))
#     for i in range(len(entities)):
#         entity_embeddings.append(mix_entity_and_description_embedding(entities[i], entity_description[i]))
#     tree = build_hierarchy(entities, entity_embeddings)
#     print(tree)
#     print('\n')
#     print(find_leaves(tree))

def map_child_to_parent(tree, parent_map=None, current_parent=None):
    #把所有的{孩子:父母}变成字典中的键值对
    if parent_map is None:
        parent_map = {}
    for key, value in tree.items():
        if current_parent is not None:  # Map current key to its parent
            parent_map[key] = current_parent
        if isinstance(value, dict):  # Recursively process the dictionary
            map_child_to_parent(value, parent_map, key)
    return parent_map

def node2parentpath(tree, source_cluster):
    #找到source_cluster直到root的路径
    #依赖map_child_to_parent
    parent_path = []
    child_parent = map_child_to_parent(tree)
    current_parent = child_parent[source_cluster]
    while current_parent in child_parent.keys():
        parent_path.append(current_parent)
        current_parent = child_parent[current_parent]
    parent_path.append(current_parent)
    return parent_path


def find_nearest_keys_lca_based(tree, input_key, parent_map, m=5):
    def _find_distance(parent_map, key, root):
        distance = 0
        while key != root:
            key = parent_map[key]
            distance += 1
        return distance

    def _find_lca(parent_map, key1, key2):
        ancestors = set()
        # Climb up from key1 to the root, collecting all ancestors
        while key1 in parent_map:
            ancestors.add(key1)
            key1 = parent_map.get(key1, None)  # Safely get parent or None if not exists
            if key1 is None:
                break
        # Climb up from key2 until we find the first common ancestor
        while key2 not in ancestors:
            key2 = parent_map.get(key2, None)  # Safely get parent or None if not exists
            if key2 is None:
                return None  # If reached the top without finding an ancestor, return None
        return key2

    def _distance_between_keys(parent_map, key1, key2):
        # Find root two levels above current key
        root1 = parent_map.get(key1)
        if root1:
            root1 = parent_map.get(root1)

        root2 = parent_map.get(key2)
        if root2:
            root2 = parent_map.get(root2)

        # Find LCA considering two levels up as the root
        if root1 and root2:
            lca = _find_lca(parent_map, key1, key2)
            if lca:
                distance1 = _find_distance(parent_map, key1, lca)
                distance2 = _find_distance(parent_map, key2, lca)
                return distance1 + distance2
        return -1  # Return -1 if no valid LCA is found

    all_keys = set(parent_map.keys())
    distances = []

    for key in all_keys:
        if key != input_key:
            dist = _distance_between_keys(parent_map, input_key, key)
            if dist != -1:  # Only consider valid distances
                distances.append((key, dist))

    # Sort the list of distances based on distance, and return the first n keys
    distances.sort(key=lambda x: x[1])

    if len(distances) < m:
        return [key for key, dist in distances]

    return [key for key, dist in distances[:m]]

# def labeling_hierarchy_to_entities(hierarchy, entity_info, num_threads=4):
#     from collections import defaultdict
#     from tqdm import tqdm
#     leaf_keys, leaf_values = find_leaves(hierarchy)
#
#     label2entity = defaultdict(list)
#     for entity in entity_info.keys():
#         label2entity[entity_info[entity]["text_label"]].append(entity)
#
#     for i in tqdm(range(len(leaf_values))):
#         entity_label_list = leaf_values[i]
#         if not isinstance(entity_label_list, list):
#             entity_label_list = [entity_label_list]
#         for entity_label in entity_label_list:
#             entities = label2entity.get(entity_label, [])
#             parent_path = node2parentpath(hierarchy, leaf_keys[i])
#             parent_map = map_child_to_parent(hierarchy)
#             nearest_clusters_lca = find_nearest_keys_lca_based(
#                 hierarchy, leaf_keys[i], parent_map, m=5
#             )
#
#             for entity in entities:
#                 entity_info[entity]["cluster"] = leaf_keys[i]
#                 entity_info[entity]["parent_path"] = parent_path
#                 entity_info[entity]["nearest_clusters_lca"] = nearest_clusters_lca
#
#     return entity_info
#
# def test_labeling_hierarchy_to_entities():
#     entity_info = create_entity_info(entities)
#     res = labeling_hierarchy_to_entities(test_tree,entity_info)
#     print(res)

