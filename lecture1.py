# %% [markdown]
# # INTUITION

# %%
import math 
import numpy as np
import matplotlib.pyplot as plt 
%matplotlib inline

# %%
def f(x):
    return 3*x**2 - 4*x + 5

# %%
f(3.0)

# %%
xs = np.arange(-5,5,0.25)
ys = f(xs)

plot=plt.plot(xs,ys)


# %% [markdown]
# # DERIVATIVE OF FUNCTION

# %%
h=0.0001
x=3.0
dfdx = ((f(x+h)-f(x))/h)
print(dfdx)

# %%
def slope(x,h=0.0001):
    return ((f(x+h)-f(x))/h)


xs = np.arange(-5,5,0.25)
ys = slope(xs)

plot=plt.plot(xs,ys)

# %%
class Value:
    def __init__(self,data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"
    def __add__(self,other):
        return Value(self.data + other.data, {self,other}, _op='+')
    def __mul__(self,other):
        return Value(self.data * other.data, {self,other}, _op='*')
    

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

d = a*b+c 

# %%
from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  nodes, edges = trace(root)
  
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ | data %.4f |  }" % ( n.data), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot

# %%
draw_dot(d)

# %%



