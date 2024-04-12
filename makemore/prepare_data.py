import torch

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
    itos = {}
    stoi = {}

    itos[0] = '.'
    stoi['.'] = 0

    letters = sorted(list(set("".join(names))))

    for idx, letter in enumerate(letters):
        stoi[letter] = idx + 1
        itos[idx + 1] = letter

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

