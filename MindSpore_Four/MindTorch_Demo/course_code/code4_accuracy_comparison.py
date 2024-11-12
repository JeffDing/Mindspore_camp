# from mindtorch.tools import mstorch_enable  # 使用mindtorch时启用
import torch
from torch import nn
from mindtorch.tools import debug_layer_info

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        print(x.shape)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork()
# torch.save(model.state_dict(), 'parameters.pth')   # 首次保存模型参数时候启用
model.load_state_dict(torch.load('parameters.pth'))  
model.eval()

# debug_layer_info(model, frame='pytorch')    # 使用pytorch时启用
debug_layer_info(model)                      # 使用mindtorch时启用

input = torch.ones((3, 28, 28))
output = model(input)
