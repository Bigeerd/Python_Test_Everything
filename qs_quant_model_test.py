def train():
    import torch
    gamma = torch.nn.Parameter(torch.Tensor([0.1]),requires_grad=True)
    loss = gamma * 2
    loss.backward()
    print(gamma.grad)

def test():
    import torch
    gamma = torch.Tensor([1.])
    g = {
        'a1' : 'b1',
        'a2' : 'b2',
        'a3' : 'b3',
    }
    for i in g.items():
        print(i)
    print(gamma.item())

def zeros():
    import torch
    x = torch.empty(1, 2, 3,device='cuda')
    print(x)
    a = torch.zeros(
        1,2,3,
        dtype=torch.float32,
        device='cuda',
        requires_grad=True,
        out=x,
    )
    print(x)

def zero1():
    import torch
    a = torch.zeros(
        1,2,3,
        dtype=torch.float32,
        requires_grad=True,
    )
    print(a)
    loss = 2 * a.sum()#loss = 2 × (a1 + a2 + a3 + a4 + a5 + a6)
    loss.backward()
    print(a.grad)#2

def uniform():
    import torch
    relation_embedding = torch.Tensor([[1,2,3],[4,5,6]])
    torch.nn.init.uniform_(
        tensor=relation_embedding,
        a = -1,
        b = 1,
    )
    print(relation_embedding)
    print(relation_embedding.shape)
    print(relation_embedding.size())

def chunk():
    import torch
    x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    print(x.chunk(3,dim=1))
    print(torch.chunk(x,3,dim=0))

def torchcat():
    import torch
    a = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    b = torch.tensor([[3,2,1],[6,5,4],[9,8,7]])
    shape1 = torch.cat([a,b],dim=1).shape
    print(f'shape1 = {shape1}')
    out1 = torch.empty(shape1)
    torch.cat([a,b],dim=1,out=out1)
    print(out1)

def torcat():
    import torch
    a = torch.randn(2,3,6)
    print(a.shape)
    print(a[:,:,:list(a.shape)[-1] // 2].shape)
    b = torch.randn(2,3)
    print(b.tolist())

def test121():
    import torch
    a = torch.randn(2,3,2)
    b = torch.randn(2,3,2)
    x_shape = torch.cat([
            a[:,:,list(a.shape)[-1] // 2],
            b[:,:,list(b.shape)[-1] // 2],
        ],dim=1).shape
    x = torch.empty(x_shape)
    torch.cat(
        [
            a[:,:,list(a.shape)[-1] // 2],
            b[:,:,list(b.shape)[-1] // 2],
        ],
        dim=1,
        out=x
    )
    print(x)

def dotsum():
    import torch
    a = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    print(a.sum(dim=0).sum())

def summ1():
    import torch
    a = torch.randn(2,3,2,requires_grad=True)
    loss = 10 * a.sum()
    loss.backward()
    print(a.grad)

def para():
    import torch
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net,self).__init__()
            self.param1 = torch.nn.Parameter(torch.randn(2,3,2))
            self.param2 = torch.nn.Parameter(torch.tensor([[1,2,3],[4,5,6],[7,8,9]],dtype=torch.float32))
    model = Net()
    for param1,param2 in model.named_parameters():
        print(param1,param2)

def tensoritem():
    import torch
    tensor = torch.tensor([[[[[[1]]]]]])
    print(tensor.item())

def exceppt():
    try:
        import torch
        torch.cat('asd')
    except Exception as e:
        print(e)
        raise ValueError("wocao!!!") from None

def norm():
    import torch
    a = torch.tensor([[3],[4]],dtype=torch.float32)
    print(torch.norm(a,p=1,dim=1))

def dosum():
    import torch
    a = torch.tensor([[[[1],[1],[1]],[[2],[2],[2]],[[3],[3],[3]]]])
    print(a.sum(dim=2))#对应维度要删掉

def catstack():
    import torch
    a = torch.randn(2,3,2)
    b = torch.randn(2,3,2)
    print(torch.cat([a,b],dim=0).shape)
    print(torch.stack([a,b],dim=2).shape)

def chunksqueeze():
    import torch
    dim = 1
    a,b,c = torch.chunk(torch.randn(2,3,2),3,dim=dim)
    print(a.shape,b.shape,c.shape)
    print(a.squeeze(list(a.shape)[dim]).shape,b.squeeze(list(b.shape)[dim]).shape,c.squeeze(list(c.shape)[dim]).shape)

def clamp():
    import torch
    a = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    print(torch.clamp(a,min=2,max=7))

def fuz():
    import torch
    a = torch.tensor([1,2.,3,4,5,6,7,8,9],dtype=torch.float32)
    b = torch.tensor([2],dtype = torch.int32)
    d = a != b
    print(d)

def unzip():
    dim = [1,2,3]
    dim1 = (1,2,3)
    print(*dim)
    print(*dim1)
    dict = {
        'a1' : 'b1',
        'a2' : 'b2',
        'a3' : 'b3',
    }
    def f1(a1,a2,a3):
        print(a1,a2,a3)
    f1(**dict)

def indexselect():
    import torch
    a = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    print(a.index_select(dim=1,index=torch.tensor([0,2])))
    a.view(9,-1)[-1] = 1
    print(a)
