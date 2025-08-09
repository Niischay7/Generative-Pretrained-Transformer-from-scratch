import numpy as np
# Building a dense layer
class Layer_Dense:
  def __init__(self,n_inputs,n_neurons):
    self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
    self.biases = np.zeros((1,n_neurons))

  def forward(self,inputs):
    self.output = np.dot(inputs,self.weights) + self.biases

inputs = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

layer1 = Layer_Dense(4, 5)
layer1.forward(inputs)

print("Layer 1 Output:")
print(layer1.output)

!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input.txt','r',encoding='utf-8') as f:
  text = f.read()

print("length of dataset in characters: " , len(text))

print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

""" Why We Do This :-
Before training a neural network on text, we must:

1.Convert characters to numbers → Neural networks only work with numbers.
2.To do that, we need a vocabulary — a mapping from:
3.Character to Integer (e.g., 'a' → 0, 'b' → 1)
4.Integer to Character (to decode later)

This step prepares us to build those mappings next."""

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hii there"))
print(decode(encode("hii there")))

""" converting the strings into integer mapping and then integer to string mapping 
this is called character level tokenisation to prepare text for a neural network as it deals only in integers"""

import torch
data = torch.tensor(encode(text), dtype = torch.long)
print(data.shape , data.dtype)
print(data[:1000])

""" here in we convert the entire dataset into a tensor by slicing it into training batches
and move it to gpu only if needed and then feed it into a model directly """

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[:n]

block_size = 8;
train_data[:block_size + 1]

x = train_data[:block_size]
y = train_data[1:block_size + 1]

for i in range(block_size):
  context = x[:i+1]
  target = y[i]
  print(f"when input is {context} the target : {target}")

""" here in we start preparing the training exaples for model 
basically teaching our model /network that when you see this equence of characers the next character should be this"""

torch.manual_seed(1337)
batch_size = 4 # we will train on 4 different sequqnces in parallel.
block_size = 8 # context length - model will look at maximum character beore predicting the next one.


def get_batch(split):
  data = train_data if split == 'train' else val_data
  # we will chose train_data or val_data depending on the split
  ix = torch.randint(len(data) - block_size,(batch_size,))

  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i + block_size+1] for i in ix])

  return x,y

xb,yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print(yb.shape)
print(yb)


for i in range(batch_size): # batch dimension
  for j in range(block_size):# time dimension
    context = xb[i,:j+1]
    target = yb[i,j]
    print(f"when input is {context.tolist()} the target : {target}")


""" Basically we feed multiple sequences into the model at once enabling faster training and give the model both short and long contexts which prepares the proper(x,y) pairs for the next token prediction"""
