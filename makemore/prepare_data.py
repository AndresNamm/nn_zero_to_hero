import torch
from dataclasses import dataclass


@dataclass
class CharacterMapping:
    stoi: dict
    itos: dict
    vocab_size: int

def generate_character_mapping(names=["John","Jane","Jim","Jill","Jack","Jenny"]) -> CharacterMapping:	    
    # build the vocabulary of characters and mappings to/from integers
    chars = sorted(list(set(''.join(names))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    vocab_size = len(itos)
    return CharacterMapping(stoi,itos,vocab_size)



def generate_training_data(names=["John","Jane","Jim","Jill","Jack","Jenny"], context_size=3,print_out=False) -> tuple:
    """
    Generate training data for a neural network model.

    Args:
        block_size (int): The size of the context block.
        print_out (bool): Whether to print the generated data.

    Returns:
        torch.Tensor: The input data (X) as a tensor.
        torch.Tensor: The target data (Y) as a tensor.
    """
    character_mapping = generate_character_mapping(names)
    
    stoi = character_mapping.stoi
    itos = character_mapping.itos

    X = []
    Y = []

    for name in names:
        name = name + "."
        context = context_size * [0]
        if print_out:
            print(name)
        for ch in name:
            context_str = "".join([itos[idx] for idx in context])
            predict_str = ch
            if print_out:
                print(f"{context_str}-->{predict_str}")
            X.append(context)
            Y.append(stoi[ch])
            context = context[1:] + [stoi[ch]]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

