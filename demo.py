lst = []

lst.extend([1,2,3] * 2)

print(lst)

exit()


import torch

cdw = torch.tensor([0.2, 1, 0.5])
vec = torch.tensor([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
])
cwd_weight = cdw.unsqueeze(-1).repeat(1, 4)
print(cwd_weight)
print(torch.mul(vec, cwd_weight))