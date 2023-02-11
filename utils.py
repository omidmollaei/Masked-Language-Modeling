
import os
import dataclasses
import tensorflow as tf
import tensorflow_text as text
from pathlib import Path
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from typing import Any, Iterable, Dict, NewType, Union, get_type_hints
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

DataClassType = NewType("DataClassType", Any)


class MainArgumentParser(ArgumentParser):
    """
    This subclass of argparse.ArgumentParser uses type hints on dataclasses to generate CLI arguments.
    """
    def __init__(self, dataclass_types: Iterable[DataClassType], **kwargs):
        """
        Args:
            dataclass_types:
                A list of dataclass types which will be parsed, and we will 'fill' instances with parsed args.
            kwargs:
                Passed to `argparse.ArgumentParser()` in the regular way.
        """
        if "formatter_class" not in kwargs:
            kwargs["formatter_class"] = ArgumentDefaultsHelpFormatter
        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = list(dataclass_types)
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    def _add_dataclass_arguments(self, dtype: DataClassType):
        parser = self
        type_hints: Dict[str, type] = get_type_hints(dtype)
        for field in dataclasses.fields(dtype):
            field.type = type_hints[field.name]
            self._parse_dataclass_field(parser, field)

    @staticmethod
    def _parse_dataclass_field(parser: ArgumentParser, field: dataclasses.Field):
        field_name = f"--{field.name}"
        kwargs = field.metadata.copy()  # field metadata is not used at all.

        aliases = kwargs.pop("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]

        if field.type is not bool:
            kwargs["type"] = field.type

            if field.default is not dataclasses.MISSING:
                kwargs["default"] = field.default
            else:
                kwargs["required"] = True
            parser.add_argument(field_name, *aliases, **kwargs)
        else:
            parser.add_argument(field_name, *aliases, action="store_true",  **kwargs)

    def parse_args_into_dataclasses(self):
        """Parse command-line args into instances of the specified dataclass types.
        Returns:
                Tuple of consisting of dataclass instances in the same order they were given to initializer.
        """
        all_args = self.parse_args()
        output_dataclasses = []
        for d_class in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(d_class)}
            inputs = {k: v for k, v in vars(all_args).items() if k in keys}

            obj = d_class(**inputs)
            output_dataclasses.append(obj)
        return tuple(output_dataclasses)


def write_vocab_file(file_path: str, vocab: list):
    with open(file_path, "w") as f:
        for token in vocab:
            print(token, file=f)


def build_tokenizer(dataset_args: DataClassType):
    # -------- Prepare train dataset to build vocabulary of tokenizer ----------- #
    dataset = load_tf_dataset(
        path=dataset_args.train_files_path,
        cycle_length=dataset_args.cycle_length,
    )
    dataset = dataset.filter(lambda line: tf.strings.length(line) > 0)  # filter empty lines
    dataset = dataset.batch(1000).prefetch(2)

    # ------------------- Generate the vocabulary --------------------------- #
    bert_tokenizer_params = dict(lower_case=dataset_args.lower_case)
    reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
    bert_vocab_args = dict(
        vocab_size=dataset_args.vocab_size - 1,  # we would add [MASK] token later
        reserved_tokens=reserved_tokens,
        bert_tokenizer_params=bert_tokenizer_params,
        leawrn_params={},
    )
    print("Generating tokenizer vocabulary ...")
    vocab = bert_vocab.bert_vocab_from_dataset(dataset, **bert_vocab_args)
    vocab.append("[MASK]")
    reserved_tokens.append("[MASK]")

    # update the size of vocabs if it is lower than specified vocab size by user in CLI.
    dataset_args.vocab_size = len(vocab)

    # save vocabs as text file in disk.
    vocab_path = os.path.join(dataset_args.tokenizer_saving_path, "vocabulary")
    os.makedirs(vocab_path, exist_ok=True)
    vocab_path = os.path.join(vocab_path, "vocab.txt")
    write_vocab_file(vocab_path, vocab)

    # ---------------------- Build the tokenizer -------------------------- #
    tokenizer = SubWordTokenizer(
        path=vocab_path,
        lower_case=
        dataset_args.lower_case
    )


class SubWordTokenizer(tf.Module):
    def __init__(self, vocab_path: str, lower_case:bool = True, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = ["[START]", "[END]", "[UNK]", "[PAD]", "[MASK]"]
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)



def load_tf_dataset(path: str, cycle_length: Union[None, int]):
    files_ds = tf.data.Dataset.list_files(os.path.join(path, "*.txt"))
    dataset = files_ds.interleave(
        map_func=lambda filepath: tf.data.TextLineDataset(filepath),
        cycle_length=cycle_length,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset
