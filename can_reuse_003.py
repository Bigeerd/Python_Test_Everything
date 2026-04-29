import pickle

from openai import OpenAI
client = OpenAI(api_key="ollama_is_free", base_url="http://localhost:11434/v1")

def gpt_AI_response(model, prompt, seed=44):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0,
        seed=seed,
    )
    return response

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
    response = gpt_AI_response(model="qwen2:0.5b", prompt=prompt)
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

def build_tree(children, n_leaves, entity_labels):
    hierarchy = {f"Leaf_{i}": entity_labels[i] for i in range(n_leaves)}
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

import ast
entities = ["Apple", "Orange", "strawberry", "Banana", "Computer", "Phone", "Airplane"]
entities_description = []
embeddings = []
for entity in entities:
    entities_description.append(generate_entity_description(entity))
for i in range(len(entities)):
    embeddings.append(mix_entity_and_description_embedding(entities[i], entities_description[i]))
print(evaluate_threshold(entities, embeddings))



