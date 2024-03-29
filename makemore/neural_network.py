from typing import Callable
from dataclasses import dataclass
import torch
import torch.nn.functional as F



@dataclass
class TrainingParams:
    iterations: int = 1000
    batch_size: int = 30
    learning_rate: Callable[[int],int] = lambda x: 1


class NeuralNetwork:


    def __init__(self, context_size:int = 3, hidden_layer_neurons: int = 100, letter_embedding_dimensions: int = 2):

        self.context_size = context_size
        self.hidden_layer_neurons = hidden_layer_neurons
        self.letter_embedding_dimensions = letter_embedding_dimensions

        self.g = torch.Generator().manual_seed(2147483647)
        self.c = torch.randn(27,self.letter_embedding_dimensions,generator=self.g) 
        self.w1 = torch.randn(self.letter_embedding_dimensions*self.context_size,self.hidden_layer_neurons,generator=self.g)
        self.b1 = torch.randn(self.hidden_layer_neurons,generator=self.g) # Add to every neuron bias
        self.w2 = torch.randn(self.hidden_layer_neurons,27,generator=self.g)
        self.b2 = torch.randn(27) # Add to every neuron bias

        print("Shape of c:", self.c.shape)
        print("Shape of w1:", self.w1.shape)
        print("Shape of b1:", self.b1.shape)
        print("Shape of w2:", self.w2.shape)
        print("Shape of b2:", self.b2.shape)

        self.params = [self.c,self.w1,self.b1,self.w2,self.b2]
        param_count = sum([p.nelement()  for p in self.params])
        print(f"Total parameters currently in NN {param_count}")
        
            

        for p in self.params:
            p.requires_grad=True


    def train(self, X,Y, training_params: TrainingParams):
        # print color green 
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


            

            if i % 500==0:
                # Calculate total loss
                total_h = (self.c[X].view(-1,self.context_size*self.letter_embedding_dimensions)  @ self.w1 + self.b1).tanh()
                total_logits = total_h @ self.w2 + self.b2
                total_loss = F.cross_entropy(total_logits, Y)

                print(f"Total loss currently is {total_loss}")
                print(f"Current learning rate is {learning_rate}")
                
            for param in self.params: 
                param.data -= param.grad * learning_rate #type: ignore



