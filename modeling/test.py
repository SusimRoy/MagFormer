import torch 

a = torch.ones((2,2,3,4))
b = torch.zeros((1,1))
c = a*b
print(c.shape)
print(c.max(),c.min())
# print(a*b)