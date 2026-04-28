
def seed_everything(seed=42):
    #in:seed
    #out:固定所有随机
    import random,os,numpy as np,torch
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed) #管HASH运算，不一定有用
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU
    torch.backends.cudnn.deterministic = True # 算法一致
    torch.backends.cudnn.benchmark = False # 不许快算法

# def create_test_json():
#     import json
#     data = []
#     for i in range(10):
#         data.append(f'keys{i+1}')
#     json.dump(data,open('json_test.json','w'),ensure_ascii=False,indent=4)

def read_json(file_path):
    #in:file_path
    #out:file_path里的内容
        import json
        return json.load(open(file_path,'r',encoding='utf-8'))

# def create_test_npy():
#     import numpy as np
#     data = np.array([[1,2,3],[4,5,6],[7,8,9]])
#     np.save('test_npy',data)

def load_npy_to_cpu_or_cuda(file_path,allow_gpu=False):
    #in:file_path,allow_gpu
    #out:把file_path里的东西,转成tensor,加载到cpu或gpu里面
    import numpy as np,torch
    data = np.load(file_path)
    data = torch.tensor(data,dtype=torch.float32)
    if allow_gpu and torch.cuda.is_available():
        data = data.cuda()
    return data

# entity2id = {"张三":0, "李四":1, "王五":2, "北京":3, "中国":4, "程序员":5}
# relation2id = {"出生":0, "国籍":1, "职业":2}
# def create_test_triple():
#     data = """
#     张三\t出生\t北京
#     李四\t国籍\t中国
#     王五\t职业\t程序员
#     """
#     with open('test_triple.txt','w') as f:
#         f.write(data)
# create_test_triple()

def read_triple(file_path,entity2id,relation2id,delimiter='\t'):
    #in:file_path,entity2id,relation2id,delimiter
    #out:转成向量的三元组
    triples = []
    with open(file_path,'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(delimiter)

            if len(parts) != 3:
                continue

            h,r,t = parts[0],parts[1],parts[2]

            triples.append((entity2id[h],relation2id[r],entity2id[t]))
    return triples

def set_logger(log_path):
    #in:log_path
    #out:设置好日志格式,发日志时写入log_path并输出在控制台上
    import logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        filename=log_path,
        filemode='a'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def int_to_str_code(x, base_chars):
    #base_chars=string.ascii_lowercase
    #in:int_to_str_code(26)
    #out:ba
    if x < 0:
        raise ValueError("x must be non-negative")
    s = ""
    base = len(base_chars)
    if x == 0:
        return base_chars[0]
    while x > 0:
        x, rem = divmod(x, base)
        s = base_chars[rem] + s
    return s

def quantized_to_token(quantized, token_prefix="<#", token_suffix=">", token_num=32):
    #rely:int_to_str_code
    #in:quantized_to_token(26,)
    #out:<#ba>
    # in:quantized_to_token([26,26,26],)
    # out:<#ba><#ba><#ba>
    if isinstance(quantized, (int, float)):
        quantized = int(quantized)
        return f"{token_prefix}{int_to_str_code(quantized)}{token_suffix}"
    elif isinstance(quantized, (list, tuple)):
        return "".join(
            [
                f"{token_prefix}{int_to_str_code(int(x))}{token_suffix}"
                for x in quantized[:token_num]
            ]
        )
    else:
        raise TypeError(f"Unsupported type: {type(quantized)}")

def str_code_to_int(s, base_chars):
    #base_chars=string.ascii_lowercase
    #in:str_code_to_int('ba')
    #out:26
    base = len(base_chars)
    x = 0
    for ch in s:
        x = x * base + base_chars.index(ch)
    return x

def token_to_quantized(token_str, token_prefix="<#", token_suffix=">"):
    #rely:str_code_to_int
    #in:token_to_quantized('[<#bmma>,<#a>]',)
    #out:[26000, 0]
    import re
    pattern = re.escape(token_prefix) + r"([A-Za-z0-9]+)" + re.escape(token_suffix)
    codes = re.findall(pattern, token_str)
    if not codes:
        raise ValueError(f"No valid tokens found in: {token_str}")
    values = [str_code_to_int(code) for code in codes]
    return values[0] if len(values) == 1 else values

def check_json_if_keys_exist(data,required_keys):
    #rely:read_json
    #in:check_json_if_keys_exist(data,required_keys)
    #out:KeyError: 'Missing required key: keys_11'
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key: {key}")
    return data


