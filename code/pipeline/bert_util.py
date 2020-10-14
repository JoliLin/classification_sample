import torch

def find_pruneable_heads_and_indices(heads, n_heads: int, head_size: int, already_prune_heads):
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads

    for head in heads:
        head= head - sum(1 if h<head else 0 for h in already_pruned_heads)
        mask[head] = 0

    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()

    return heads, index

def prune_linear_layer(layer:torch.nn.Linear, index:torch.LongTensor, dim:int=0) -> torch.nn.Linear:
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()

    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer_bias.requires_grad = True
    return new_layer

