import numpy as np
import torch,os,logging,argparse
import matplotlib.pyplot as plt
# import gymnasium


def exp_np_newaxis():# 实验1：np.newaxis 加维度（1维 → 3维）
    print("===== 实验1：np.newaxis 加维度 =====")
    x_np = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)
    print("原始形状:", x_np.shape)  # (10,)

    # 加维度
    x_new = x_np[np.newaxis, :, np.newaxis]
    print("加完形状:", x_new.shape)  # (1, 10, 1)
    print()

def exp_reshape():# 实验2：reshape 改变形状（等价 newaxis）
    print("===== 实验2：reshape 改变形状 =====")
    x_np = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)
    print("原始形状:", x_np.shape)

    # reshape 成 1×10×1
    x_reshaped = x_np.reshape(1, 10, 1)
    print("reshape 后形状:", x_reshaped.shape)
    print()

def exp_torch_from_numpy():# 实验3：torch.from_numpy 转换 numpy → tensor
    print("===== 实验3：torch.from_numpy 转换 =====")
    x_np = np.array([1, 2, 3], dtype=np.float32)
    x_tensor = torch.from_numpy(x_np)
    print("numpy 类型:", type(x_np))
    print("tensor 类型:", type(x_tensor))
    print("形状不变:", x_tensor.shape)
    print()

def exp_flatten():# 实验4：flatten 展平（1×10×1 → 10）
    print("===== 实验4：flatten 展平 =====")
    a = np.random.rand(1, 10, 1)
    print("原始形状:", a.shape)
    b = a.flatten()
    print("展平形状:", b.shape)
    print()

def exp_torch_stack():# 实验5：torch.stack 拼接
    print("===== 实验5：torch.stack 拼接 =====")
    a = torch.tensor([1.1])
    b = torch.tensor([2.2])
    c = torch.tensor([3.3])
    outs = [a, b, c]
    result = torch.stack(outs, dim=1)
    print("拼接后形状:", result.shape)
    print(result)


def subplots():
    fig, ax = plt.subplots(2, 5)
    print(fig,ax)

def makeenv():
    env = gymnasium.make('CartPole-v1')
    print("环境创建成功：", env)
    print("环境类型：", type(env))
    env = env.unwrapped
    print("解包后：", env)
    print("可以访问内部变量啦！")
    print("小车最大位置 x_threshold =", env.x_threshold)
    print("杆子最大角度 theta_threshold_radians =", env.theta_threshold_radians)
    N_ACTIONS = env.action_space.n
    print("动作数量 N_ACTIONS =", N_ACTIONS)  # 输出 2
    print("动作空间：", env.action_space)  # 输出 Discrete(2)
    N_STATES = env.observation_space.shape[0]
    print("状态维度 N_STATES =", N_STATES)  # 输出 4
    print("状态空间：", env.observation_space)
    print("状态形状：", env.observation_space.shape)
    print("随机一个动作：", env.action_space.sample())  # 0 或 1
    print("动作类型：", type(env.action_space.sample()))  # <class 'int'>
    print("是不是int类型：", isinstance(env.action_space.sample(), int))  # True

    # 所以最终
    ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
    print("ENV_A_SHAPE =", ENV_A_SHAPE)  # 输出 0

def weight():
    # 实验7：线性层 + 权重初始化
    fc1 = torch.nn.Linear(4, 50)  # 输入4，输出50

    print("fc1 结构：", fc1)
    print("fc1.weight 形状：", fc1.weight.shape)  # (50,4)
    print("初始化前权重（前5个）：")
    print(fc1.weight.data[:5])

    # 正态分布初始化
    fc1.weight.data.normal_(0, 0.1)
    print("初始化后权重（前5个）：")
    print(fc1.weight.data[:5])

def uni():
    print(np.random.uniform(0,999))

def test_unsqueeze():
    # 原始状态：4个数（CartPole的状态）
    x = np.array([0.1, 0.2, 0.3, 0.4])
    print("原始 x shape:", x.shape)  # (4,)

    # 转tensor
    x_tensor = torch.FloatTensor(x)

    # 在 0 维度加一维
    x_out = torch.unsqueeze(x_tensor, 0)
    print("unsqueeze 后 shape:", x_out.shape)  # (1,4)
    print("加一维后内容:\n", x_out)

    x_out = torch.unsqueeze(x_tensor, 1)
    print("unsqueeze 后 shape:", x_out.shape)  # (1,4)
    print("加一维后内容:\n", x_out)

def test_torch_max():
    # 模拟网络输出：两个动作的Q值
    actions_value = torch.tensor([[3.2, 5.7],[4.0,5.0]])
    print("网络输出Q值:\n", actions_value)

    # dim=1 按行取最大
    max_val, max_idx = torch.max(actions_value, dim=1)

    print("最大值分数:", max_val)
    print("最大索引（动作）:", max_idx)


def test_randint():
    N_ACTIONS = 20
    for _ in range(10):
        a = np.random.randint(0, N_ACTIONS)
        print("随机动作:", a)

def vstack():
    # 造两个很小的数组
    a = np.array([1, 2, 3])   # 一行
    b = np.array([4, 5, 6])   # 另一行

    # vstack = vertical stack = 竖着叠起来
    c = np.vstack((a, b))

    print("a：")
    print(a)
    print("b：")
    print(b)
    print("竖着叠完 c：")
    print(c)
    print("形状:", c.shape)


def sik():
    # 方法1：列表推导式（原代码用的）
    result1 = [np.linspace(-1,1,3) for _ in range(2)]
    print("方法1：")
    print(result1)

    # 方法2：普通for循环（你说的写法）
    result2 = []
    for i in range(2):
        result2.append(np.linspace(-1,1,3))

    print("\n方法2：")
    print(result2)

    # 你会发现：完全一样！

def uniform():
    # 从 1～2 之间，随机生成 3 个数
    a = np.random.uniform(low=1, high=2, size=3)
    print(a)

def new_axis():
    a = np.array([1, 2, 3])  # shape (3,)
    b = a[:, np.newaxis]  # shape (3,1)

    print("原来：", a.shape)
    print("加维度后：", b.shape)

def power():
    x = np.array([-2, -1, 0, 1, 2])
    y = np.power(x, 2)  # 平方
    print(y)  # [4,1,0,1,4]

def randn():
    # 生成 2行3列 的随机数
    a = torch.randn(2, 3)
    print(a)
    print(a.shape)  # torch.Size([2, 3])

def cat():
    a = torch.tensor([[1, 2], [3, 4]])  # (2,2)
    b = torch.tensor([[5], [6]])  # (2,1)

    # 按第1维拼接（往右拼）
    c = torch.cat((a, b), dim=1)

    print(c)
    print(c.shape)  # (2, 3)

def log():#就是数学里的 ln
    x = torch.tensor([0.5, 0.9])
    y = torch.log(x)
    print(y)

def broadcast():
    paint_points = np.vstack([np.linspace(-1,1,15) for _ in range(64)])
    a = np.random.uniform(1,2,size=64)[:,np.newaxis]
    print(paint_points.shape)
    print(a.shape)
    print((paint_points*a).shape)
    print((a*paint_points).shape)

    a = np.array([[2], [3]])  # (2,1)
    b = np.array([[1, 1], [2, 2]])  # (2,2)

    print(a * b)  # 逐元素（广播+哈达玛）
    # 结果：
    # [[2 2]
    #  [6 6]]

def float():
    np_data = np.array([1, 2, 3], dtype=np.float64)  # numpy 默认双精度
    print("原始numpy类型:", np_data.dtype)  # float64

    # ----------------------
    # 写法1：.float() —— 最常用！
    # ----------------------
    t1 = torch.from_numpy(np_data).float()
    print("写法1 .float() 类型:", t1.dtype)  # torch.float32

    # ----------------------
    # 写法2：dtype=torch.float32 —— 创建时指定
    # ----------------------
    t2 = torch.tensor(np_data, dtype=torch.float32)
    print("写法2 dtype=float32 类型:", t2.dtype)  # torch.float32

    # ----------------------
    # 写法3：torch.FloatTensor() —— 旧版构造器
    # ----------------------
    t3 = torch.FloatTensor(np_data)
    print("写法3 FloatTensor 类型:", t3.dtype)  # torch.float32

def detach():
    # 一个带梯度的张量
    x = torch.tensor([2.], requires_grad=True)

    y = x * 3
    print(y.requires_grad)  # True → 还连着计算图
    print(x.requires_grad)

    # 一 detach，梯度断了！
    y_detach = y.detach()
    print(y_detach.requires_grad)  # False
    print(x.requires_grad)

def randint():
    print(np.random.randint(1,4))

def shape():
    a = torch.randn(1,1)
    print(f'a.shape : {a.shape}')
    outs = []
    for i in range(64):
        outs.append(a)
    print(outs)
    b = torch.stack(outs, dim=1)
    print(b.shape)


# 实验代码：triples vs set_triples 对比
def test_triples_and_set_triples():
    # 1. 模拟原始triples（列表，有重复）
    triples = [
        ("猫", "属于", "哺乳动物"),
        ("狗", "属于", "哺乳动物"),
        ("猫", "吃", "鱼"),
        ("猫", "属于", "哺乳动物")  # 重复的三元组
    ]
    print("===== 1. 原始triples（列表） =====")
    print(f"内容：{triples}")
    print(f"长度：{len(triples)}（包含重复）")
    print(f"是否有序：是（按输入顺序）\n")

    # 2. 转成set_triples（集合，去重）
    # 注意：列表里的元组可以直接转集合，自动去重
    set_triples = set(triples)
    print("===== 2. 转成set_triples（集合） =====")
    print(f"内容：{set_triples}")
    print(f"长度：{len(set_triples)}（已去重）")
    print(f"是否有序：否（集合无序）\n")

    # 3. 核心用法：快速判断“某个三元组是否是正确样本”
    print("===== 3. 核心用法：快速校验 =====")
    # 正确的三元组（在集合里）
    correct_triple = ("猫", "属于", "哺乳动物")
    # 错误的三元组（不在集合里）
    wrong_triple = ("猫", "属于", "鱼类")

    # 用set_triples判断，速度极快（O(1)）
    if correct_triple in set_triples:
        print(f"✅ {correct_triple} 是正确样本，不能当负样本！")
    if wrong_triple not in set_triples:
        print(f"❌ {wrong_triple} 是错误样本，可以当负样本！")


def count_frequency(triples, start=4):
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

# 第一步：真实的文字三元组
# -------------
triples_text = [
    ("张三", "国籍", "中国"),
    ("张三", "国籍", "美国"),
    ("爱因斯坦", "出生地", "乌尔姆"),
]

# -------------
# 第二步：给文字编身份证（数字ID）
# -------------
entity2id = {
    "张三": 0,
    "中国": 1,
    "美国": 2,
    "爱因斯坦": 3,
    "乌尔姆": 4,
}

id2entity = {
    0: "张三",
    1: "中国",
    2: "美国",
    3: "爱因斯坦",
    4: "乌尔姆",
}

relation2id = {
    "国籍": 0,
    "出生地": 1,
}

id2relation = {
    0 : "国籍",
    1 : "出生地",
}

# # 转成数字三元组
# triples_id = []
# for h, r, t in triples_text:
#     triples_id.append((entity2id[h], relation2id[r], entity2id[t]))
#
# print("文字三元组：")
# for t in triples_text:
#     print(t)
#
# print("\n数字三元组（给模型看的）：")
# for t in triples_id:
#     print(t)
#     print(t[0],t[1],t[2])
#
# result = count_frequency(triples_id)
# print("\ncount_frequency 结果：")
# for k, v in result.items():
#     print(k, "→", v)

def get_true_head_and_tail(triples):
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

# a,b = get_true_head_and_tail(triples_id)
# print(a,b)
# for k,v in a.items():
#     print(id2entity[v[0]],id2relation[k[0]],id2entity[k[1]])
#
# print(triples_id[1])
def randintsize():
    print(np.random.randint(20,size=10))

def npin1d():
    # 测试数据
    A = [1, 2, 3, 4, 5]  # 随机生成的候选数字
    B = [2, 4]  # 标准答案

    # 正确函数：np.isin()
    res = np.isin(A, B)
    print("原始判断（在不在B里）：")
    print(res)

    # 加上 invert=True（反过来：不在里面才是True）
    res_invert = np.isin(A, B, invert=True)
    print("\n加 invert=True 后（我们代码要用的）：")
    print(res_invert)

def mask():
    # 转成 numpy 数组！
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = np.array([True, False, False, False, False, False, True, True, True])

    # 现在过滤！只保留 True 位置的数
    result = a[b]

    print("过滤前：", a)
    print("mask：", b)
    print("过滤后（只留True）：", result)

def concatenate():
    a = [
        [2, 5, 7],
        [1, 9, 1],
        [3, 6, 8]
    ]
    print(np.concatenate(a))
    print(np.concatenate(a)[:3])
    a = torch.Tensor(a)
    print(a.flatten())

def catstack():
    w1 = torch.Tensor([0.5])  # shape=(1,)
    w2 = torch.Tensor([0.8])  # shape=(1,)
    return torch.stack([w1, w2], dim=0),torch.cat([w1, w2], dim=0)

def loop():
    print([_ for _ in range(10)])

def make():
    # ---------------
    # 1. 构造 args 对象
    # ---------------
    class Args:
        def __init__(self):
            self.data_path = "./test_data"  # 数据文件夹

    args = Args()

    # ---------------
    # 2. 自动创建测试文件（代码帮你建）
    # ---------------
    os.makedirs(args.data_path, exist_ok=True)

    # 写 entities.dict
    with open(os.path.join(args.data_path, "entities.dict"), "w") as f:
        f.write("0\t小明\n")
        f.write("1\t小红\n")
        f.write("2\t学校\n")
        f.write("3\t北京\n")

    # 写 relations.dict
    with open(os.path.join(args.data_path, "relations.dict"), "w") as f:
        f.write("0\t喜欢\n")
        f.write("1\t在...里\n")
        f.write("2\t朋友\n")

def split_and_strip():
    # ----------------------
    # 测试 1：strip() 作用
    # 功能：去掉 两头 的空格、换行、制表符
    # ----------------------
    print("===== 测试 strip() =====")

    s1 = "  我是带空格的字符串  "
    s2 = "\n\t小明\t\n"  # 有换行、制表符

    print("原字符串:", s1)
    print("strip()后:", s1.strip())

    print("原字符串:", s2)
    print("strip()后:", s2.strip())

    # ----------------------
    # 测试 2：split() 作用
    # 功能：按指定符号 切开字符串 → 变成列表
    # ----------------------
    print("\n===== 测试 split() =====")

    line1 = "0\t小明"  # \t 是制表符
    line2 = "101,102,103"  # 逗号分隔

    # 默认按空格切
    print("按空格切:".ljust(15), "a b c".split())

    # 按制表符切（对应你的 entities.dict）
    print("按制表符切:".ljust(15), line1.split("\t"))

    # 按逗号切
    print("按逗号切:".ljust(15), line2.split(","))

def makedirs():
    path = os.path.join("./a","b")
    print(path)
    os.makedirs('./a', exist_ok=True)
    with open(path,"a") as f:
        f.write("aaa\n")

def show_loggings():
    logging.basicConfig(level=logging.INFO)
    logging.info("我是日志！！\n")

def argparser():
    # 1. 空盒子
    parser = argparse.ArgumentParser()

    # 2. 加参数规则
    parser.add_argument("--data_path", type=str, default="data")

    # 3. 解析 → 变成 args
    args = parser.parse_args()

    # 4. 直接用！
    print(args.data_path)

