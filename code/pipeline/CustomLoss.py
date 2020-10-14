import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
	def __init__(self, gamma=0, alpha=None, size_average=True):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
		if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
		if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
		self.size_average = size_average

	def forward(self, input, target):
		if input.dim()>2:
			input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
			input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
			input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
		target = target.view(-1,1)

		logpt = F.log_softmax(input, dim=1)
		logpt = logpt.gather(1,target)
		logpt = logpt.view(-1)
		pt = Variable(logpt.data.exp())

		if self.alpha is not None:
			if self.alpha.type()!=input.data.type():
				self.alpha = self.alpha.type_as(input.data)
			at = self.alpha.gather(0,target.data.view(-1))
			logpt = logpt * Variable(at)

		loss = -1 * (1-pt)**self.gamma * logpt
		if self.size_average: return loss.mean()
		else: return loss.sum()

class ContrastiveLoss(nn.Module):
	def __init__(self, margin):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin
		self.eps = 1e-9

	def forward(self, emb1, emb2, target, size_average=True):
		distance = (emb1-emb2).pow(2).sum(1)
		loss = 0.5*(target.float()*distance +( 1 + -1*target).float() * F.relu(self.margin-(distance+self.eps).sqrt()).pow(2))
		if size_average:
			return loss.mean()
		else:
			return loss.sum()

class LabelSmoothingLoss(nn.Module):
	def __init__(self, classes=2, smoothing=0.00001, dim=-1):
		super(LabelSmoothingLoss, self).__init__()
		self.confidence = 1.0 - smoothing
		self.smoothing = smoothing
		self.cls = classes
		self.dim = dim

	def forward(self, pred, target):
		pred = pred.log_softmax(dim=self.dim)
		with torch.no_grad():
			# true_dist = pred.data.clone()
			true_dist = torch.zeros_like(pred)
			true_dist.fill_(self.smoothing / (self.cls - 1))
			true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
		return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class cluster(nn.Module):
        def __init__(self, cluster_number=2, dim=64):
            super(cluster, self).__init__()

            init_centers = torch.zeros(2, 64, dtype=torch.float)
            nn.init.xavier_uniform_(init_centers)
            self.cluster_centers = nn.Parameter(init_centers)

        def forward(self, emb):
            norm_squared = torch.sum((emb.unsqueeze(1)-self.cluster_centers)**2, 2)
            numerator = 1.0 / (1.0 +(norm_squared))
            print(numerator/torch.sum(numerator, dim=1, keepdim=True))
            exit()

