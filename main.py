"""
Main script to train BERT. Run this script from command line with required arguments.
"""


import os
import utils
import tensorflow as tf
from model import mlm_model
from typing import Optional, Dict
from dataclasses import dataclass, field


@dataclass
class _ReservedTokens:
    padding: str = field(default="[PAD]")
    unknown: str = field(default="[UNK]")
    start: str = field(default="[START]")
    end: str = field(default="[END]")
    mask: str = field(default="[MASK]")


@dataclass
class ModelArguments:
    """ Bert config. The model architecture can be fully customized by these arguments. The default values are
    mostly same as original BERT Base model."""
    model_name: Optional[str] = field(
        default="model",
        metadata={
            "help": "A optional Name for model",
        }
    )
    model_dim: Optional[int] = field(
        default=768,
        metadata={
            "help": "Model dimension (Embedding and positional embedding dimension too). "
        }
    )
    num_attention_heads: Optional[int] = field(
        default=12,
        metadata={
            "help": "Number of attention heads for each layers in model."
        }
    )
    num_hidden_layers: Optional[int] = field(
        default=12,
        metadata={
            "help": "Number of hidden layers in model."
        }
    )
    intermediate_dense_size: Optional[int] = field(
        default=3000,
        metadata={
            "help": "Number of units in intermediate dense layer."
        }
    )
    classification_units: Optional[int] = field(
        default=0,
        metadata={
            "help": "If set to a value more than 0, the model would have two outputs. One for MLM task and one"
                    "for classification on the vector related to first token's output. This field indicates the"
                    "number of units in classifier layer."
        }
    )
    dropout_rate: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "Dropout rate inside the model to avoid over-fitting."
        }
    )
    activation: Optional[str] = field(
        default="relu",
        metadata={
            "help": "Activation function of hidden layers inside the model."
        }
    )
    vocab_size: Optional[int] = field(
        default=35_000,
        metadata={
            "help": "Size of input vocabulary to the model."
        }
    )
    layer_norm_eps: Optional[float] = field(
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
    lower_case: bool = field(
        default=False,
        metadata={
            "help": "Whether to lowercase the dataset or not"
        }
    )
    cycle_length: Optional[int] = field(
        default=5,
        metadata={
            "help": "The number of input elements that will be processed concurrently while reading text files."
        }
    )
    tokenizer_saving_path: Optional[str] = field(
        default=os.path.join("./", "tokenizer"),
        metadata={
            "help": "An optional path to save the tokenizer in. Default is in current path."
        }
    )
    reserved_tokens: _ReservedTokens = field(
        default=_ReservedTokens(),
        metadata={
            "help": "An instance of another dataclass which holds all the special tokens in one place."
        }
    )
    overwrite_vocab: bool = field(
        default=False,
        metadata={
            "help": "If set to True and a vocab file exists in tokenizer saving path (indicated by "
                    "`tokenizer_saving_path` arg), a new vocab file will be built and replace the old one."
                    "If no vocab file found, a new vocab file will be built."
        }
    )
    overwrite_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "If set to True, If there is already a pretrained tokenizer in tokenizer saving path (indicated"
                    "by `tokenizer_saving_path` arg), a new tokenizer will be built and replaced by that. If set to"
                    "False, the already pretrained tokenizer will be loaded (if exists)."
        }
    )


def main():
    parser = utils.MainArgumentParser((ModelArguments, TrainingDataArguments))
    model_args, dataset_args = parser.parse_args_into_dataclasses()
    dataset_args.vocab_size = model_args.vocab_size
    init_vocab_size = model_args.vocab_size

    # -- Build tokenizer
    print("Building tokenizer ... ", end="")
    tokenizer = utils.build_tokenizer(dataset_args)
    model_args.vocab_size = dataset_args.vocab_size
    tokenizer.save()
    print(f"done!. [Saved to: {dataset_args.tokenizer_saving_path}]")
    print(f"[Number of vocab size updated from {init_vocab_size} to {model_args.vocab_size}]")

    # -- Build model
    print("Building model ... ", end="")
    model = mlm_model(params=model_args)
    num_params = model.count_params()
    print(f"done! [Total number of params: {num_params}]")


if __name__ == "__main__":
    main()
