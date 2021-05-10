import argparse


def get_args() -> dict:
    parser = argparse.ArgumentParser(description='Naive Bayes Attention Masking for Offensive Language Detection')

    # input parameters
    parser.add_argument("-al", "--apply-likelihood", type=bool, required=False, default=True)
    parser.add_argument("-fn", "--formula-name", type=str, required=False, default="add_likelihood")
    parser.add_argument("-ml", "--max-len", type=int, required=False, default=64)
    parser.add_argument("-nc", "--n-classes", type=int, required=False, default=2)

    # training hyperparameters
    parser.add_argument("-mn", "--model-name", type=str, required=True)
    parser.add_argument("-bs", "--batch-size", type=int, required=True)
    parser.add_argument("-lr", "--learning-rate", type=float, required=True)
    parser.add_argument("-ep", "--num-epochs", type=int, required=True)
    
    # Common parameters between models
    parser.add_argument("-bt", "--bert-trainable", type=bool, required=True)
    parser.add_argument("-lg", "--logits", type=bool, required=False, default=True)
    parser.add_argument("-avg", "--average", type=str, required=False, default="macro")

    # NULI parameters
    parser.add_argument("-nd", "--nuli-dropout", type=float, required=False, default=0.3)

    # Kungfupanda parameters
    parser.add_argument("-hs", "--hidden-size", type=int, required=False, default=150)
    parser.add_argument("-lf", "--linear-in-features", type=int, required=False, default=300)
    parser.add_argument("-kd", "--kungfupanda-dropout", type=float, required=False, default=0.1)

    # KUSAIL parameters
    parser.add_argument("-nf", "--num-filters", type=int, required=False, default=32)
    parser.add_argument("-fs", "--filter-sizes", type=list, required=False, default=[1, 2, 3, 4, 5])
    parser.add_argument("-ksd", "--kusail-dropout", type=float, required=False, default=0.3)
    
    args = vars(parser.parse_args())
    return args