import torch
import mindtorch.torch as mtorch
import mindspore as ms

# 1. Print learning rate
torch_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(2.0))], lr=0.01)
print("torch lr is {}".format(torch_optimizer.param_groups[0]['lr']))

mtorch_optimizer = mtorch.optim.SGD([mtorch.nn.Parameter(mtorch.tensor(2.0))], lr=0.01)
print("mindtorch lr no float is {}".format(mtorch_optimizer.param_groups[0]['lr']))
print("mindtorch lr float is {}".format(float(mtorch_optimizer.param_groups[0]['lr'])))


# 2. Modified learning rate
torch_optimizer.param_groups[0]['lr'] = 0.1
print("modified torch lr is {}".format(torch_optimizer.param_groups[0]['lr']))

ms.set_context(mode=ms.context.PYNATIVE_MODE)
mtorch_optimizer.param_groups[0]['lr'] = 0.1
print("PYNATIVE_MODE modified mindtorch lr is {}".format(mtorch_optimizer.param_groups[0]['lr']))

ms.set_context(mode=ms.context.GRAPH_MODE)
mtorch_optimizer = mtorch.optim.SGD([mtorch.nn.Parameter(mtorch.tensor(2.0))], lr=0.01)
ms.ops.assign(mtorch_optimizer.param_groups[0]['lr'], 0.2)
print("GRAPH_MODE modified mindtorch lr is {}".format(torch_optimizer.param_groups[0]['lr']))

# 3. Custom optimizer
class TRanger(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.5, k=6):
        defaults = dict(lr=lr, alpha=alpha)
        super().__init__(params, defaults)
        self.k = k
    def __setstate__(self, state):
        print("set state called")
        super().__setstate__(state)
    def step(self, closure=None):
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                p_data_fp32 = p.data.float()
                state = self.state[p]
                state['step'] += 1
                p_data_fp32.add_(grad)
                p.data.copy_(p_data_fp32)
        return loss

tranger = TRanger([torch.nn.Parameter(torch.tensor(2.0))], lr=0.01)
print("Init TRanger", tranger)

class MTRanger(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.5, k=6):
        defaults = dict(lr=lr, alpha=alpha)
        super().__init__(params, defaults)
        self.k = k
    def __setstate__(self, state):
        print("set state called")
        super().__setstate__(state)
    def step(self, grads, closure=None): # 需要新增grads作为函数入参，以便传入梯度
        loss = None
        i = -1                           # 声明一个索引，用来遍历grads入参
        for group in self.param_groups:
            for p in group['params']:
                i = i + 1                # 索引递增
                grad = grads[i]          # grad从入参中获取。如果对应Parameter没有参与求导，grad为0
                p_data_fp32 = p.data.float()
                state = self.state[p]
                state['step'] += 1
                p_data_fp32.add_(grad)
                p.data.copy_(p_data_fp32)
        return loss
    
mtranger = MTRanger([torch.nn.Parameter(torch.tensor(2.0))], lr=0.01)
print("Init MTRanger", mtranger)
