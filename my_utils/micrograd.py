from graphviz import Digraph
import math

class Value:

    def __init__(self,data, _children=(), _op='',label=''):
        self.data:float = data
        self.grad:float = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._gradient_updates=0

    def __repr__(self):
        return f"Value(data={self.data})"

    def get_node_label(self):
            
            label = f"{self.label}={self.data}" if self.label else self.data
            return "{" + f"{label} | grad={self.grad}" + " | " + f"grad_updates={self._gradient_updates}" + "}"
    
    def __add__(self,other): 
        other = other if isinstance(other,Value) else Value(other)
        out =  Value(self.data + other.data, {self,other}, _op='+')

        def _backward(): # This backward function is called from the out variable, aka the result value (parent), however the _backward function assigns gradients to input variables (children) leveraging its own gradient # 
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
            self._gradient_updates+=1
            other._gradient_updates+=1
        out._backward = _backward
    
        return out 
    
    def __radd__(self,other):# https://stackoverflow.com/questions/9126766/addition-between-classes-using-radd-method
        return self+other

    def __mul__(self,other): 
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, {self,other}, _op='*')

        def _backward():
            # We are adding to gradient because the value could have effect
            # through multiple later stages. 
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad
            self._gradient_updates+=1
            other._gradient_updates+=1

        out._backward = _backward   
        return out
    
    def __pow__(self,other):
        assert isinstance(other,(int,float))

        out = Value(self.data**other,{self,},_op=f'**{other}')

        def _backward():
            self.grad += other * self.data ** (other-1) * out.grad
            self._gradient_updates+=1

        out._backward = _backward

        return out   
    
    def __neg__(self):
        return self * (-1)

    def __sub__(self,other):
        return self + (-other)
    
    def __truediv__(self,other):
        return self * other**-1

    def __rmul__(self,other): # https://stackoverflow.com/questions/5181320/under-what-circumstances-are-rmul-called
        return self * other

    def tanh(self): # https://wikimedia.org/api/rest_v1/media/math/render/svg/b8dc4c309a551cafc2ce5c883c924ecd87664b0f
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad
            self._gradient_updates+=1

        out._backward = _backward     
        return out
    
    def exp(self):
        x=self.data
        out = Value(math.exp(x), (self,),'exp')
        def _backward():
            self.grad += math.exp(x) * out.grad
            self._gradient_updates+=1

        out._backward=_backward
        return out 
        
    
    def backward(self): 
        visited=set()
        topo=[]
        def build_topo(o):
            if o not in visited:
                visited.add(o)
                for node in o._prev:
                    build_topo(node)
                topo.append(o)
        self.grad=1
        build_topo(self)

        for node in reversed(topo):
            node._backward()




def trace(root):
  # builds a set of all nodes and edges in a graph. Runs a DFS from the answer to all the components
  # that contributed to it.
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
    #dot.node(name = uid, label = "{ %s | data %.4f |  }" % ( n.label ,n.data), shape='record')
    dot.node(name = uid, label = n.get_node_label(), shape='record')

    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot


import random

class Neuron:
    def __init__(self,nin,layer_idx,neuron_idx):
        self.w = [Value(random.uniform(-1, 1),label=f'W_{layer_idx}{neuron_idx}{parameter_idx+1}') for parameter_idx in range(nin)]
        self.b = Value(random.uniform(-1,1),label=f'B_{layer_idx}{neuron_idx}')
    
    def __call__(self,x): # Returns scalar value between -1 and 1
        n = sum((xi*wi for xi,wi in zip(x,self.w)),self.b); n.label = 'n'
        tan = n.tanh(); tan.label = 'tanh'
        return tan
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:

    def __init__(self, nin, nout, layer_idx): # Input is how many inputs each neuron takes. It's the number of dimensions in
        # n-1 layer, output is how many neurons will be generated. 
        self.neurons = []
        for neuron_idx in range(nout):
            self.neurons.append(Neuron(nin,layer_idx,neuron_idx+1))

    def __call__(self,x):
        out = []
        for neuron in self.neurons:
            out.append(neuron(x))

        return out[0] if len(out)==1 else out
    
    def parameters(self):
        return [parameter for neuron in self.neurons for parameter in neuron.parameters() ]#

class MLP:
    def __init__(self, nin, layers):
        sz = [nin]+layers
        self.layers = []
        for i in range(1, len(sz)): # generate neurons for each layer. We start from 1 not 0 index because the 0th index describes size of data not neural network layer. It's needed to determine firs layer neuron size.  
            # Each neuron in layer sz[i-1] inputs 
            # Each layer has sz[i] neurons 
            self.layers.append(Layer(sz[i-1],sz[i],i))


    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x

    def parameters(self):
        return [parameter for layer in self.layers for parameter in layer.parameters()]#

    def represent(self):
        for idx,layer in enumerate(self.layers):
            print(f"Layer: {idx}")
            print(f"Has {len(layer.neurons)} neurons")
            print(f"Each neuron has {len(layer.neurons[0].w)+1} inputs")
            print()
            