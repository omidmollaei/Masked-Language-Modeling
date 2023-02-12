
import os
import dataclasses
import tensorflow as tf
import tensorflow_text as text
from pathlib import Path
from argparse import ArgumentParser, ArgumentTypeError
from argparse import ArgumentDefaultsHelpFormatter
from typing import Any, Iterable, Dict, NewType, Union, Optional, get_type_hints
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

DataClassType = NewType("DataClassType", Any)


def string_to_bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif s.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {s} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


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

        origin_type = getattr(field.type, "__origin__", field.type)
        if origin_type is Union:
            if str not in field.type.__args__ and (
                    len(field.type.__args__) != 2 or type(None) not in field.type.__args__):
                raise ValueError(
                    "Only `Union[X, NoneType]` (i.e., `Optional[X]`) is allowed for `Union` because"
                    " the argument parser only supports one type per argument."
                    f" Problem encountered in field '{field.name}'."
                )
            if type(None) not in field.type.__args__:
                # filter `str` in Union
                field.type = field.type.__args__[0] if field.type.__args__[1] == str else field.type.__args__[1]
                origin_type = getattr(field.type, "__origin__", field.type)
            elif bool not in field.type.__args__:
                # filter `NoneType` in Union (except for `Union[bool, NoneType]`)
                field.type = (
                    field.type.__args__[0] if isinstance(None, field.type.__args__[1]) else field.type.__args__[1]
                )
                origin_type = getattr(field.type, "__origin__", field.type)

        if field.type is bool or field.type == Optional[bool]:
            kwargs["type"] = string_to_bool
            if field.type is bool or (field.default is not None and field.default is not dataclasses.MISSING):
                # Default value is False if we have no default when of type bool.
                default = False if field.default is dataclasses.MISSING else field.default
                # This is the value that will get picked if we don't include --field_name in any way
                kwargs["default"] = default
                # This tells argparse we accept 0 or 1 value after --field_name
                kwargs["nargs"] = "?"
                # This is the value that will get picked if we do --field_name (without value)
                kwargs["const"] = True
        else:
            kwargs["type"] = field.type
            if field.default is not dataclasses.MISSING:
                kwargs["default"] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                kwargs["default"] = field.default_factory()
            else:
                kwargs["required"] = True
        parser.add_argument(field_name, *aliases, **kwargs)


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
    def __init__(self, vocab_path: str, lower_case: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = ["[START]", "[END]", "[UNK]", "[PAD]", "[MASK]"]
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)

        self.start_token_id = tf.argmax(tf.constant(self._reserved_tokens) == "[START]")
        self.end_token_id = tf.argmax(tf.constant(self._reserved_tokens) == "[END]")

        # create signature for export:
        self.tokenize.get_concrete_function(tf.TensorSpec(shape=[None], dtype=tf.string))
        self.add_start_end.get_concrete_function(tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))
        self.cleanup_text.get_concrete_function(tf.TensorSpec(shape=[None], dtype=tf.string))
        self.detokenize.get_concrete_function(tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

    # docstring must be added to below functions
    @tf.function
    def add_start_end(self, tokenized):
        count = tokenized.bounding_shape()[0]
        starts = tf.fill([count, 1], self.start_token_id)
        ends = tf.fill([count, 1], self.end_token_id)
        return tf.concat([starts, tokenized, ends], axis=1)

    @tf.function
    def cleanup_text(self, sentences):
        pattern = b"\\[START\\]\\[END\\]\\[PAD\\]"
        sentences_cleaned = tf.strings.regex_replace(sentences, pattern, "")
        return tf.strings.strip(sentences_cleaned)

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)   # produce a ragged tensor (int64)
        enc = enc.merge_dim(-2, -1)              # merge the `word` and `word-piece` axes.
        return self.add_start_end(enc)

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        sentences = tf.strings.reduce_join(words, separator=" ")
        return self.cleanup_text(sentences)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)


def load_tf_dataset(path: str, cycle_length: Union[None, int]):
    files_ds = tf.data.Dataset.list_files(os.path.join(path, "*.txt"))
    dataset = files_ds.interleave(
        map_func=lambda filepath: tf.data.TextLineDataset(filepath),
        cycle_length=cycle_length,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset
