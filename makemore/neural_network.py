from typing import Callable
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor
import enum


class InitializationType(enum.Enum):
    no_fixes = "no fixes"
    avoid_being_confidently_wrong = "avoid being confidently_wrong"
    squash_h = "squash h"




@dataclass
class TrainingParams:
    iterations: int = 1000
    batch_size: int = 30
    learning_rate: Callable[[int],float] = lambda x: 1.0


class NeuralNetwork:


    def __init__(self, context_size:int = 3, hidden_layer_neurons: int = 100, letter_embedding_dimensions: int = 2, print_flag=True, generator_seed=2147483647, initialization_type: InitializationType = InitializationType.squash_h):

        self.context_size = context_size
        self.hidden_layer_neurons = hidden_layer_neurons
        self.letter_embedding_dimensions = letter_embedding_dimensions
        self.name = f"NN_{context_size}_{hidden_layer_neurons}_{letter_embedding_dimensions}"   

        self.g = torch.Generator().manual_seed(generator_seed)
        self.c = torch.randn(27,self.letter_embedding_dimensions,generator=self.g) 


        if initialization_type == InitializationType.no_fixes:

            self.w1 = torch.randn(self.letter_embedding_dimensions*self.context_size,self.hidden_layer_neurons,generator=self.g)
            self.b1 = torch.randn(self.hidden_layer_neurons,generator=self.g) # Add to every neuron bias
            self.w2 = torch.randn(self.hidden_layer_neurons,27,generator=self.g) # 
            self.b2 = torch.randn(27,generator=self.g) # Add to every neuron bias 
        elif initialization_type == InitializationType.avoid_being_confidently_wrong:

            self.w1 = torch.randn(self.letter_embedding_dimensions*self.context_size,self.hidden_layer_neurons,generator=self.g)
            self.b1 = torch.randn(self.hidden_layer_neurons,generator=self.g) # Add to every neuron bias
            self.w2 = torch.randn(self.hidden_layer_neurons,27,generator=self.g) * 0.01 # make each weight smaller so we would not be confidently wrong.
            self.b2 = torch.zeros(27) # Add to every neuron bias -- set it initialy to 0 to again avoid being confidently wrong.
        elif initialization_type == InitializationType.squash_h:
            self.w1 = torch.randn(self.letter_embedding_dimensions*self.context_size,self.hidden_layer_neurons,generator=self.g) * 0.2
            self.b1 = torch.randn(self.hidden_layer_neurons,generator=self.g) * 0.01 
            # Changing w1 and b1 to smaller numbers avoids h1 becoming too large which intself would make tanh go to 1 which would make
            # the specific neuron to not be able to learn based on some training examples.
            self.w2 = torch.randn(self.hidden_layer_neurons,27,generator=self.g) * 0.01 # make each weight smaller so we would not be confidently wrong.
            self.b2 = torch.zeros(27) # Add to every neuron bias -- set it initialy to 0 to again avoid being confidently wrong.
        else:
            print("else")

        self.params = [self.c,self.w1,self.b1,self.w2,self.b2]
        
        # meta flags
        self.losses = []
        self.loss_iterations=[]
        self.total_iterations = 0
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

    @staticmethod 
    # https://en.wikipedia.org/wiki/Softmax_function
    def softmax(logits):
        counts=logits.exp()
        return counts/counts.sum()

    def generate_name(self):
        context_idx = self.context_size * [0]
        predicted_idx=-1
        name=[]
        while predicted_idx!=0:
            logits: Tensor = ((self.c[torch.Tensor(context_idx).int()].view(-1,self.context_size*self.letter_embedding_dimensions) @ self.w1 + self.b1).tanh() @ self.w2 + self.b2)
            probabilities_manual = NeuralNetwork.softmax(logits) 
            probabilities = torch.softmax(logits,1) # Runs softmax column level
            assert probabilities.allclose(probabilities_manual)
            # Choose with some probability the next letter

            predicted_idx=torch.multinomial(probabilities.view(-1),1).item()
            context_idx = context_idx[1:] + [predicted_idx]
            name.append(predicted_idx)
        return name

    def calculate_loss(self, X,Y):
        return F.cross_entropy((self.c[X].view(-1,self.context_size*self.letter_embedding_dimensions) @ self.w1 + self.b1 ).tanh() @ self.w2 + self.b2,Y)

    def train(self, X,Y, training_params: TrainingParams=TrainingParams(iterations=1000, batch_size=30, learning_rate=lambda x: 0.1)):
        # Check X
        assert X.shape[1] == self.context_size
        # Check Y
        assert len(Y.shape) == 1
        # print color green 
        if self.print_flag:
            print("\033[92m" + f"Start training" + "\033[0m")
        for i in range(training_params.iterations):
            self.total_iterations += 1
            minibatch = torch.randint(0,X.shape[0],(training_params.batch_size,))          
            h = (self.c[X[minibatch]].view(-1, self.context_size * self.letter_embedding_dimensions) @ self.w1 + self.b1).tanh()
            logits = h @ self.w2 + self.b2
            loss = F.cross_entropy(logits, Y[minibatch])

            for param in self.params:
                param.grad=None

            loss.backward()

            learning_rate = training_params.learning_rate(i)

            if i % int(training_params.iterations/3)==0:
                # Calculate total loss
                total_h = (self.c[X].view(-1,self.context_size*self.letter_embedding_dimensions)  @ self.w1 + self.b1).tanh()
                total_logits = total_h @ self.w2 + self.b2
                total_loss = F.cross_entropy(total_logits, Y)

                if self.print_flag:
                    print(f"Loss after {self.total_iterations} epochs: {total_loss}")

            self.losses.append(loss.item())
            self.loss_iterations.append(self.total_iterations)

            for param in self.params: 
                param.data -= param.grad * learning_rate #type: ignore

        if self.print_flag:
            print(f"{self.name} loss:after {training_params.iterations} epochs: {self.losses[-1]}")



