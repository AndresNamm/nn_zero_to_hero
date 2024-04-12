from makemore.prepare_data import generate_training_data
from makemore.neural_network import NeuralNetwork, TrainingParams
import argparse


def main():

    # read input args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--context_size", type=int, default=3)
    parser.add_argument("-hl","--hidden_layer_neurons", type=int, default=100)
    parser.add_argument("-le","--letter_embedding_dimensions", type=int, default=2, help="Number of dimensions for letter embeddings")
    parser.add_argument("-i","--iterations", type=int, default=1000)
    parser.add_argument("-b","--batch_size", type=int, default=30)
    args = parser.parse_args()


    context_size = args.context_size
    hidden_layer_neurons = args.hidden_layer_neurons
    letter_embedding_dimensions = args.letter_embedding_dimensions
    iterations = args.iterations
    batch_size = args.batch_size

    with open('data/names.txt', encoding='utf-8') as f:
        names = f.readlines()
        names = [name.strip() for name in names]


    X,Y = generate_training_data(names,context_size,False)
    # print shape of X and Y
    print("Shape of X:", X.shape)
    print("Shape of Y:", Y.shape)
    nn = NeuralNetwork(context_size,hidden_layer_neurons,letter_embedding_dimensions)

    training_params = TrainingParams(iterations,batch_size)
    nn.train(X,Y,training_params)


main()