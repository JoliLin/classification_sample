import torch.nn 
import torch
import math

class GELU(torch.nn.Module):
	def forward(self, x):
		return 0.5 * x *( 1 +torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * torch.pow(x, 3))) )
