from utils import tab_printer
from parser import parameter_parser
from capsgnn import CapsGNNTrainer
import torch
import pickle as pkl
def main():
    """
    Parsing command line parameters, processing graphs, fitting a CapsGNN.
    """
    args = parameter_parser()
    tab_printer(args)
    model = CapsGNNTrainer(args)
    model.fit()
    # torch.save(model.model,'./pretrained_50.mod')
    #with open('./pretrained_50.mod','wb') as rf:
    #    pkl.dump(model,rf)
    model.score()
    model.save_predictions()

if __name__ == "__main__":
    main()
