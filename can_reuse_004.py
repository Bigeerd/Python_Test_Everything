from can_reuse_003 import get_AI_response
import json
NAME_CLUSTER_PROMPT = '''Given the following entities from a cluster: {entities}.
Provide a name for these entities which can describe them uniformly as a cluster.
Name: '''
UPDATE_TWO_CLUSTERS_PROMPT = '''Given the cluster A '{cluster_name_1}': [{entities_1}];
and the cluster B '{cluster_name_2}': [{entities_2}].

Analyze two clusters and their entities and determine the update mode:
Update Mode 1 - Create New Cluster C: these two clusters cannot be merged, and no cluster belongs to any other.
Update Mode 2 - Merge Cluster A and B: these two clusters can be merged. The name of two clusters should be similar and entities from two clusters should be similar.
Update Mode 3 - Cluster A Covers Cluster B: cluster B belongs to cluster A. cluster B is a subcluster of cluster A. The name of cluster A should uniformly describe the entities from cluster A and the name of cluster B.
Update Mode 4 - Cluster B Covers Cluster A: cluster A belongs to cluster B. cluster A is a subcluster of cluster B. The name of cluster B should uniformly describe the entities from cluster B and the name of cluster A.

You need to select a update mode based on two clusters.
If you select mode 1, you should also suggest a name of new cluster. The new cluster name should uniformly describe two clusters.
If you select mode 2, you should suggest a name of merged cluster. The new name should be similar to cluster A and B.

Example:
Cluster A 'Thermal Insulators': [cork, fiberglass, foam];
Cluster B 'Electrical Conductors': [copper, aluminum, gold];
Select Mode 1.

Cluster A 'Sedans': [Toyota Camry, Honda Accord, Ford Fusion];
Cluster B 'SUVs': [Honda CR-V, Toyota RAV4, Ford Escape];
Select Mode 2.

Cluster A 'Feline Species': [lions, tigers, cheetahs];
Cluster B 'House Cats': [Siamese, Persian, Maine Coon];
Select Mode 3.

Cluster A 'Leafy Vegetables': [lettuce, spinach, kale];
Cluster B 'Root Vegetables': [carrots, potatoes, beets];
Select Mode 4.


Provide the output in the following JSON format:
```json
{{
    "update_mode": 1 or 2 or 3 or 4,
    "name": "merged cluster name or new cluster name"
}}
```

Output: '''
SPLIT_CLUSTER_PROMPT = '''Given entities from the cluster '{cluster_name}'.

    Analyze the entities and determine if they can be grouped into distinct and meaningful sub-clusters based on their characteristics, themes, or genres.
    If sub-clusters can be formed, provide a clear and concise name for each sub-cluster that represents the common attribute of its entities.
    Each sub-cluster should be given a new name that uniformly describes its entities. There needs to be differentiation in the names of different clusters.
    The number of sub-clusters should be two.
    If the entities are already well-grouped and don't require further sub-clustering, simply provide the original cluster.


    Provide the output in the following JSON format:
    ```json
    {{
        "Sub-cluster 1 Name": ["Entity 1", "Entity 2", ...],
        "Sub-cluster 2 Name": ["Entity 3", "Entity 4", ...],
    }}
    ```

    Example:
    Cluster: {split_example_cluster_name}
    Entities: {split_example_entities}
    Output: {split_example_subclusters}

    Cluster: {cluster_name}
    Entities: {entities}
    Output: '''
split_example_cluster_name = "Movies"
split_example_entities = ', '.join(
    ["The Godfather", "The Shawshank Redemption", "The Dark Knight", "Forrest Gump", "Inception", "The Matrix"])
split_example_subclusters = json.dumps({
    "Drama": ["The Godfather", "The Shawshank Redemption", "Forrest Gump"],
    "Action": ["The Dark Knight", "Inception", "The Matrix"],
})
cluster_name = "Fruits"
entities = ["Apple", "Banana", "Orange", "Watermelon", "Grape", "Strawberry"]
model = "qwen2.5-coder:7b"
# cluster1_name = "Fruits"
# cluster1_entities = ["Apple", "Banana", "Orange", "Grape", "Mango"]
#
# cluster2_name = "Vegetables"
# cluster2_entities = ["Carrot", "Tomato", "Potato", "Cabbage", "Lettuce"]
cluster1_name = "Cars"
cluster1_entities = ["Toyota", "BMW", "Tesla", "Ford"]

cluster2_name = "Books"
cluster2_entities = ["Novel", "Magazine", "Dictionary", "Comic"]

def find_json(text):
    import json,re
    pattern = r"```json\n([\s\S]+?)\n```"
    matched_json = re.search(pattern, text)
    if matched_json:
        extracted_json = matched_json.group(1)
        return json.loads(extracted_json)
    else:
        pattern = r"\{.*?\}"
        matched_json = re.search(pattern, text, re.DOTALL)
        if matched_json:
            extracted_json = matched_json.group()
            return json.loads(extracted_json)
        else:
            raise ValueError('No JSON structure found.')



def llm_split_cluster(cluster_name, entities, model, seed):
    entities_str = ', '.join(entities)
    while True:
        try:
            llm_output = get_AI_response(prompt=SPLIT_CLUSTER_PROMPT.format(
                cluster_name=cluster_name,
                entities=entities_str,
                split_example_cluster_name=split_example_cluster_name,
                split_example_entities=split_example_entities,
                split_example_subclusters=split_example_subclusters
            ), model=model, seed=seed)
            clusters = find_json(llm_output)
            break
        except Exception as e:
            # print('Call LLM Error.')
            # print(e)
            # continue
            print('Split failure.')
            # we assume it is hard for llms to split, so return the original cluster
            return {cluster_name: entities}
    return clusters

def test_llm_split_cluster():
    print(llm_split_cluster(cluster_name=cluster_name,entities=entities,model=model,seed=44))

def llm_update_two_clusters(cluster_name_1, entities_1, cluster_name_2, entities_2, model, seed):
    entities_1 = [entity for entity in entities_1 if entity is not None][:20]
    entities_2 = [entity for entity in entities_2 if entity is not None][:20]
    entities_str_1 = ', '.join(entities_1)
    entities_str_2 = ', '.join(entities_2)
    while True:
        try:
            llm_output = get_AI_response(prompt=UPDATE_TWO_CLUSTERS_PROMPT.format(
                cluster_name_1=cluster_name_1,
                entities_1=entities_str_1,
                cluster_name_2=cluster_name_2,
                entities_2=entities_str_2
            ), model=model, seed=seed)
            structured_output = find_json(llm_output)

            update_mode = structured_output['update_mode']
            name = structured_output['name']
            break
        except Exception as e:
            print('Call LLM Error.')
            print(e)
            continue
    return update_mode, name

def test_llm_update_two_clusters():
    dict = llm_split_cluster(cluster_name=cluster_name,entities=entities,model=model,seed=44)
    print(dict)
    cluster_name_1 = list(dict.keys())[0]
    cluster_name_2 = list(dict.keys())[1]
    entities_1 = list(dict.values())[0]
    entities_2 = list(dict.values())[1]
    print(llm_update_two_clusters(cluster_name_1, entities_1, cluster_name_2, entities_2, model, seed=44))
    print(llm_update_two_clusters(cluster1_name,cluster1_entities,cluster2_name,cluster2_entities, model, seed=44))

def llm_name_cluster(entities, model, seed):
    entities_str = ', '.join(entities)
    name = get_AI_response(prompt=NAME_CLUSTER_PROMPT.format(entities=entities_str), model=model, seed=seed)
    return name

MIN_ENTITIES_IN_LEAF = 2
MAX_ENTITIES_IN_LEAF = 30


def construct_bot_hierarchy(initial_hierarchy, model, seed):
    current_cluster_id = 0

    def recursion_construct_bot_hierarchy(root):
        nonlocal current_cluster_id, model, seed

        # root 是实体列表：["Apple", "Banana", ...]
        if isinstance(root, list):
            # 终止条件：实体太少 or 太多，停止拆分
            if len(root) < MIN_ENTITIES_IN_LEAF or len(root) > MAX_ENTITIES_IN_LEAF:
                return

            cluster_entities = root
            cluster_name = llm_name_cluster(cluster_entities, model, seed)

            # LLM 拆分出 2 个子集群
            splitted_clusters = llm_split_cluster(cluster_name, cluster_entities, model, seed)

            # 拆不成 → 退出
            if len(splitted_clusters) == 1:
                return

            # ==============================================
            # 核心修复：把列表 → 字典结构（正确写法）
            # ==============================================
            new_sub_clusters = {}
            for name, entities in splitted_clusters.items():
                cid = f'Cluster_llm_bot_{current_cluster_id}'
                new_sub_clusters[cid] = entities
                current_cluster_id += 1

            # 清空原列表，把新字典装进去
            root[:] = []
            root.append(new_sub_clusters)

            # 递归继续拆子节点
            for sub_entities in new_sub_clusters.values():
                recursion_construct_bot_hierarchy(sub_entities)

        else:
            # root 是字典，遍历继续递归
            for key, subcluster in root.items():
                recursion_construct_bot_hierarchy(subcluster)

    recursion_construct_bot_hierarchy(initial_hierarchy)
    return initial_hierarchy

test_initial_hierarchy = {
  'Fruits': ['Apple', 'Banana', 'Orange', 'Watermelon', 'Grape', 'Strawberry']
}
print(test_initial_hierarchy)
print(construct_bot_hierarchy(test_initial_hierarchy,model=model,seed=44))