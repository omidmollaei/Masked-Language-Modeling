"""
Main script to train BERT. Run this script from command line with required arguments.
"""


import os
from dataclasses import dataclass, field
from typing import Optional
from arg_parser import MainArgumentParser


@dataclass
class ModelArguments:
    """ Bert config. The model architecture can be fully customized by these arguments. The default values are
    mostly same as original BERT Base model."""
    model_name: str = field(
        default="model",
        metadata={
            "help": "A Name for model",
        }
    )
    model_dim: int = field(
        default=768,
        metadata={
            "help": "Model dimension (Embedding and positional embedding dimension too). "
        }
    )
    num_attention_heads: int = field(
        default=12,
        metadata={
            "help": "Number of attention heads for each layers in model."
        }
    )
    num_hidden_layers: int = field(
        default=12,
        metadata={
            "help": "Number of hidden layers in model."
        }
    )
    dropout_rate: float = field(
        default=0.1,
        metadata={
            "help": "Dropout rate inside the model to avoid over-fitting."
        }
    )
    activation: str = field(
        default="relu",
        metadata={
            "help": "Activation function of hidden layers inside the model."
        }
    )
    vocab_size: int = field(
        default=35_000,
        metadata={
            "help": "Size of input vocabulary to the model."
        }
    )
    layer_norm_eps: float = field(
        default=1e-6,
        metadata={
            "help": "Epsilon value of normalizer layer."
        }
    )


@dataclass
class TrainingDataArguments:
    """The argument for building a train and a validation dataset."""
    train_files_path: str = field(
        metadata={
            "help": "Path to training data files. The files must be in `.txt` format."
        }
    )
    valid_files_path: str = field(
        metadata={
            "help": "Path to validating data files. The files must be in `.txt` format."
        }
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        }
    )


def main():
    parser = MainArgumentParser((ModelArguments, TrainingDataArguments))
    model_args, dataset_args = parser.parse_args_into_dataclasses()
    print(model_args)
    print()
    print(dataset_args)


if __name__ == "__main__":
    main()


