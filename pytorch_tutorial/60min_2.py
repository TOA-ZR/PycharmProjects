import torch


a = torch.randn(2, 2)
a = ((a*3)/(a-1))
b = (a*a).sum()
print(b.grad_fn)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a*a).sum()
print(b.grad_fn)


x = torch.ones(2,2)

y = x + 2
print(y)

x = torch.ones(2, 2, requires_grad=True) #  it starts to track all operations on it,
# 开始有grad_fn
print(x)

y = x + 2
print(y)

print(x.grad_fn) # 起始tensor，没有gradient function
print(y.grad_fn)

z = y * y * 3
print(z)
out = z.mean()
print(out)
print(x.grad)
#Gradien
  # call .backgrad(), have all gradients
out.backward(torch.tensor(1.0)) # out.backward() 两个一个意思
print(x.grad)  # The gradient for this tensor will be accumulated into .grad attribute


# vector-Jacobian product
x = torch.randn(3, requires_grad=True)
y = x*2
print('y:', y)
while y.data.norm()<1000:  # L2 Norm
    print(y.data.norm())
    y = y*2
    print(y)

# y.backward()
# grad can be implicitly created only for scalar outputs
# 所计算的梯度都是结果变量关于创建变量的梯度
print(x.grad)
print(y.data)
print(y.grad_fn)

# want the vector-Jacobian product, pass the vector to backward as argument
# 必须输入和output相同size的tensor作为gradient的参数才行。
v = torch.tensor([1, 1.0, 10]) # input 1就是正常的导数
y.backward(v)
print(x.grad)

print(x.requires_grad)
print((x**2).requires_grad)
with torch.no_grad():
    print((x**2).requires_grad)

x = x.new_ones(5, 3, 2,dtype=torch.double)      # new_* methods take in sizes
print(x.size()[1:]) # get the size