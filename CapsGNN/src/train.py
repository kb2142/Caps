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
    # torch.save(model.model.state_dict(), './pretrained'+str(args.epochs)+'.mod')
    # model.model=torch.load('./pretrained_50.mod')
    model.score()
    model.save_predictions()

if __name__ == "__main__":
    main()
