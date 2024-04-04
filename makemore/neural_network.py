from typing import Callable
from dataclasses import dataclass
import torch
import torch.nn.functional as F



@dataclass
class TrainingParams:
    iterations: int = 1000
    batch_size: int = 30
    learning_rate: Callable[[int],float] = lambda x: 1.0



class NeuralNetwork:


    def __init__(self, context_size:int = 3, hidden_layer_neurons: int = 100, letter_embedding_dimensions: int = 2, print_flag=True):



        self.context_size = context_size
        self.hidden_layer_neurons = hidden_layer_neurons
        self.letter_embedding_dimensions = letter_embedding_dimensions
        self.name = f"NN_{context_size}_{hidden_layer_neurons}_{letter_embedding_dimensions}"   

        self.g = torch.Generator().manual_seed(2147483647)
        self.c = torch.randn(27,self.letter_embedding_dimensions,generator=self.g) 
        self.w1 = torch.randn(self.letter_embedding_dimensions*self.context_size,self.hidden_layer_neurons,generator=self.g)
        self.b1 = torch.randn(self.hidden_layer_neurons,generator=self.g) # Add to every neuron bias
        self.w2 = torch.randn(self.hidden_layer_neurons,27,generator=self.g)
        self.b2 = torch.randn(27) # Add to every neuron bias

        self.params = [self.c,self.w1,self.b1,self.w2,self.b2]
        
        # meta flags
        self.losses = []
        self.print_flag = print_flag

        
        
        
        param_count = sum([p.nelement()  for p in self.params])
        


        if self.print_flag:
            print("Shape of c:", self.c.shape)
            print("Shape of w1:", self.w1.shape)
            print("Shape of b1:", self.b1.shape)
            print("Shape of w2:", self.w2.shape)
            print("Shape of b2:", self.b2.shape)

            print(f"Total parameters currently in NN {param_count}")
        
        for p in self.params:
            p.requires_grad=True





    def train(self, X,Y, training_params: TrainingParams):
        # print color green 
        if self.print_flag:
            print("\033[92m" + f"Start training" + "\033[0m")
        for i in range(training_params.iterations):
            minibatch = torch.randint(0,X.shape[0],(training_params.batch_size,))          
            h = (self.c[X[minibatch]].view(-1, self.context_size * self.letter_embedding_dimensions) @ self.w1 + self.b1).tanh()
            logits = h @ self.w2 + self.b2
            loss = F.cross_entropy(logits, Y[minibatch])

            for param in self.params:
                param.grad=None

            loss.backward()

            learning_rate = training_params.learning_rate(i)

            if i % 100==0:
                # Calculate total loss
                total_h = (self.c[X].view(-1,self.context_size*self.letter_embedding_dimensions)  @ self.w1 + self.b1).tanh()
                total_logits = total_h @ self.w2 + self.b2
                total_loss = F.cross_entropy(total_logits, Y)
                self.losses.append(total_loss)


                
            for param in self.params: 
                param.data -= param.grad * learning_rate #type: ignore

        if self.print_flag:
            print(f"{self.name} loss:after {training_params.iterations} epochs: {self.losses[-1]}")