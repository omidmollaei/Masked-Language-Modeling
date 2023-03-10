
import os
import re
import dataclasses

import tensorflow as tf
import tensorflow_text as text
from pathlib import Path
from argparse import ArgumentParser, ArgumentTypeError
from argparse import ArgumentDefaultsHelpFormatter
from typing import Any, Iterable, Dict, NewType, Union, Optional, List, get_type_hints
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

DataClassType = NewType("DataClassType", Any)
cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


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
        self.dataclass_types: List[DataClassType] = list(dataclass_types)
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    def _add_dataclass_arguments(self, dtype: DataClassType):
        parser = self
        type_hints: Dict[str, type] = get_type_hints(dtype)
        for field in dataclasses.fields(dtype):
            if dataclasses.is_dataclass(field.default):  # we do not want to parse dataclass types into CLI args.
                continue
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
    """
    Save the vocabulary extracted by trained (sub-word)tokenizer to disk.
    Args:
        file_path: Path to save the file . (it should contain the file name too. e.g ./vocab/vocab.txt)
        vocab: List of vocab.
    Return:
        None.
    """
    with open(file_path, "w") as f:
        for token in vocab:
            print(token, file=f)


def load_tf_dataset(path: str, cycle_length: Union[None, int]):
    """
    Load a tensorflow dataset from text files by interleaving the lines of files.
    Args:
        path: path to the folder containing `.txt` files.
        cycle_length: Number of the files, reading concurrently.
                    (more info: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave)
    Return:
        A tf.data.Dataset with no preprocessing.
    """
    files_ds = tf.data.Dataset.list_files(os.path.join(path, "*.txt"))
    dataset = files_ds.interleave(
        map_func=lambda filepath: tf.data.TextLineDataset(filepath),
        cycle_length=cycle_length,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset


def _build_vocab_file(dataset_args: DataClassType, update_vocab_size: bool = True):
    """
    Generates a tf dataset and use tensorflow_texts methods to build a source vocab
    for sub-word tokenizer.
    Args:
        dataset_args: A namespace(an instance of a dataclass) containing the arguments for preparing the dataset.
    Returns:
        None
    """
    # ======= Generate tf-dataset ======== #
    dataset = load_tf_dataset(
        path=dataset_args.train_files_path,
        cycle_length=dataset_args.cycle_length,
    )
    dataset = dataset.filter(lambda line: tf.strings.length(line) > 0)  # filter empty lines
    dataset = dataset.batch(1000).prefetch(2)
    # ==================================== #

    # ========= Extract vocab  =========== #
    bert_tokenizer_params = dict(lower_case=dataset_args.lower_case)
    reserved_tokens = [getattr(dataset_args.reserved_tokens, f.name) for f in
                       dataclasses.fields(dataset_args.reserved_tokens) if f.name != "mask"]

    bert_vocab_args = dict(
        vocab_size=dataset_args.vocab_size - 1,  # we would add mask token later.
        reserved_tokens=reserved_tokens,
        bert_tokenizer_params=bert_tokenizer_params,
        learn_params={},
    )
    vocab = bert_vocab.bert_vocab_from_dataset(dataset, **bert_vocab_args)
    # ================================== #

    # ====== Write vocab to Disk ======= #
    vocab.append(dataset_args.reserved_tokens.mask)  # we must be sure that the mask token is the last token.
    vocab_file = os.path.join(dataset_args.tokenizer_saving_path, "vocabulary", "vocab.txt")
    write_vocab_file(vocab_file, vocab)

    if update_vocab_size:
        dataset_args.vocab_size = len(vocab)
    # ================================= #


def _build_tokenizer_from_scratch(dataset_args: DataClassType):
    """
    This function builds tokenizer from scratch. If no tokenizer is found, or `overwrite_tokenizer`
    arg is set to True by user, this function will build tokenizer.
    Args:
        dataset_args: A namespace(an instance of a dataclass) containing the arguments for preparing the dataset.
    Returns:
        An instance of `SubWordTokenizer`.
    """

    vocab_file = os.path.join(dataset_args.tokenizer_saving_path,  "vocabulary", "vocab.txt")
    vocab_file_found = True if os.path.exists(vocab_file) else False

    # ===========  Vocab File  =========== #
    if dataset_args.overwrite_vocab:
        # build vocab source file from scratch
        print(f"Generating vocab from source dataset ... ")
        _build_vocab_file(dataset_args, update_vocab_size=True)

    elif not vocab_file_found:
        # user tends to use already saved vocab source, but vocab file not found
        print(f"No vocab file found. Generating vocab from source dataset ...")
        _build_vocab_file(dataset_args, update_vocab_size=True)

    else:
        # user tends to use already saved vocab source and vocab file exists
        print(f"Loading vocab from: {vocab_file}")
        with open(vocab_file, 'r') as v:
            vocab_loaded = v.readlines()
            dataset_args.vocab_size = len(vocab_loaded)
    # ================================== #

    # =======  Build tokenizer ======== #
    tokenizer = SubWordTokenizer(
        vocab_path=vocab_file,
        saving_path=dataset_args.tokenizer_saving_path,
        reserved_tokens=dataset_args.reserved_tokens,
        lower_case=dataset_args.lower_case,
    )
    # ================================== #

    return tokenizer


def build_tokenizer(dataset_args: DataClassType):
    """
    Build a (sub-word) tokenizer from a text-dataset or load the saved tokenizer. The tokenizer
    is an instance of `SubWordTokenizer` class.
    Args:
        dataset_args: A namespace(an instance of a dataclass) containing the arguments for preparing the dataset.
    Return:
        An instance of `SubWordTokenizer`.
    """
    if not os.path.exists(dataset_args.tokenizer_saving_path):
        os.makedirs(dataset_args.tokenizer_saving_path)
        os.makedirs(os.path.join(dataset_args.tokenizer_saving_path, "vocabulary"))

    # ==== Try loading a pre-trained tokenizer from specified
    #      path by user in CLI. (`saving_tokenizer_path` arg)    ==== #
    try:
        tokenizer = load_tokenizer(dataset_args.tokenizer_saving_path)
        tokenizer_found = True
    except OSError:
        tokenizer_found = False
    # ============================================ #

    # ====== Building tokenizer ====== #
    if dataset_args.overwrite_tokenizer:
        # here we do not care if a pretrained tokenizer exists or not.
        # user wants to build one from scratch anyway.
        print("Building tokenizer from scratch ... ")
        tokenizer = _build_tokenizer_from_scratch(dataset_args)

    elif not tokenizer_found:
        # user wants to use pretrained tokenizer, but no pretrained tokenizer found.
        # so build one from scratch.
        print(f"No pretrained tokenizer found. Building one from scratch ... ")
        tokenizer = _build_tokenizer_from_scratch(dataset_args)

    else:
        # user wants to load a pretrained tokenizer and one found.
        # so we load that tokenizer.
        print(f"Loading already saved tokenizer from : {dataset_args.tokenizer_saving_path}")
        tokenizer = load_tokenizer(dataset_args.tokenizer_saving_path)
    # =============================== #

    return tokenizer


class SubWordTokenizer(tf.Module):
    def __init__(self, vocab_path: str, saving_path: str, reserved_tokens: DataClassType,
                 lower_case: bool = True, **kwargs):
        """
        Main class for model tokenizer. It's a subclass of tf.Module so we can export and reuse it easily later.
        To save/export and load an instance of this class, use tf.saved_model.(save|load)
        Args:
            vocab_path: path to vocab file which must be generated by `build_tokenizer` function before making an
                 instance from this class. The path must include the file name too.
                (e.g ./tokenizer/vocabulary/vocab.txt)
            saving_path: a path to save the (sub-word) tokenizer in.
            reserved_tokens: A dataclass instance(i.e. namespace) which maps the special/reserved tokens name into
                their value (i.e. special token itself)
            lower_case: whether to lower case the text or not.
            **kwargs: other keywords argument, passed to parent's constructor.
        """
        super().__init__(**kwargs)
        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=lower_case)
        self._reserved_tokens = reserved_tokens
        self._reserved_tokens_list = [getattr(reserved_tokens, f.name) for f in dataclasses.fields(reserved_tokens)]
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)
        self.saving_path = saving_path

        self.start_token_id = tf.argmax(tf.constant(self.vocab == self._reserved_tokens.start))
        self.end_token_id = tf.argmax(tf.constant(self.vocab == self._reserved_tokens.end))
        self.pad_token_id = tf.argmax(tf.constant(self.vocab == self._reserved_tokens.padding))
        self.mask_token_id = tf.argmax(tf.constant(self.vocab == self._reserved_tokens.mask))

        # create signature for export:
        self.tokenize.get_concrete_function(tf.TensorSpec(shape=[None], dtype=tf.string))
        self.add_start_end.get_concrete_function(tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))
        self.cleanup_text.get_concrete_function(tf.TensorSpec(shape=[None], dtype=tf.string))
        self.detokenize.get_concrete_function(tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))
        self.get_reserved_tokens.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_vocab_size.get_concrete_function()
        # self.save.get_concrete_function()

    # docstring must be added to below functions
    @tf.function
    def add_start_end(self, tokenized):
        """Add start and end token ids into each instance of input batch. Input and output are both ragged."""
        count = tokenized.bounding_shape()[0]
        starts = tf.fill([count, 1], self.start_token_id)
        ends = tf.fill([count, 1], self.end_token_id)
        return tf.concat([starts, tokenized, ends], axis=1)

    @tf.function
    def cleanup_text(self, sentences):
        """Remove reserved/special tokens from a text."""
        # pattern = b"\\[START\\]\\[END\\]\\[PAD\\]"   # this is completely depends on reserved tokens.
        bad_tokens = [re.escape(tok) for tok in self._reserved_tokens_list]
        bad_tokens_re = "|".join(bad_tokens)
        sentences_cleaned = tf.strings.regex_replace(sentences, bad_tokens_re, "")
        return tf.strings.strip(sentences_cleaned)

    @tf.function
    def tokenize(self, strings):
        """
        Tokenize a batch of strings tensor.
        Args:
            strings: string tensor, can be batched or not.
        Return:
              tokenized input string tensor (ids) with start and end tokens (ids).
        """
        enc = self.tokenizer.tokenize(strings)   # produce a ragged tensor (int64)
        enc = enc.merge_dims(-2, -1)              # merge the `word` and `word-piece` axes.
        return self.add_start_end(enc)

    @tf.function
    def detokenize(self, tokenized):
        """Converts token ids into the text. It removes the reserved tokens."""
        words = self.tokenizer.detokenize(tokenized)
        sentences = tf.strings.reduce_join(words, separator=" ", axis=-1)
        return self.cleanup_text(sentences)

    @tf.function
    def lookup(self, token_ids):
        """Converts token ids into their corresponding text word by word. (i.e. no text tokens merge)."""
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        """Returns the vocabulary size of the tokenizer.(it is different from the one specified by user in CLI.)"""
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        """Returns the path to vocabulary file (`.txt` file)"""
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        """Returns the list of reserved/special tokens"""
        return tf.constant(self._reserved_tokens_list)

    def save(self):
        tf.saved_model.save(self, self.saving_path)


def load_tokenizer(path: str):
    """
    Load a pretrained exported tokenizer from disk.
    Args:
        path: path to saved tokenizer.
    Returns: Sub-word tokenizer.
    """
    reloaded_tokenizer = tf.saved_model.load(path)
    return reloaded_tokenizer


def mask_tokenized_batch(inputs: tf.Tensor, dataset_args: DataClassType, tokenizer: SubWordTokenizer):
    """
    This function applies masking mechanism to input tensor batch.
    Args:
         inputs: input tensor batch (examples shape: [64, 512])
         dataset_args: a namespace (dataclass) contains dataset arguments
         tokenizer: tokenizer used to tokenize the strings.
    Returns:
        masked input with a certain probability with desired output
    """

    mask = tf.random.uniform(minval=0, maxval=1, shape=tf.shape(inputs)) < dataset_args.mlm_probability
    reserved_tokens_mask = inputs > 3   # mask tokens: 0, 1, 2
    mask = tf.math.logical_and(mask, reserved_tokens_mask)
    mask = tf.fill(tf.shape(mask), tokenizer.mask_token_id) * tf.cast(mask, tf.int64)
    masked_inputs = tf.reduce_max([inputs, mask], axis=0)
    outputs_mask = tf.math.equal(masked_inputs, tokenizer.mask_token_id)
    outputs = inputs * tf.cast(outputs_mask, tf.int64)
    neg_outputs = tf.cast(tf.fill(tf.shape(masked_inputs), -100), dtype=tf.int64)
    neg_outputs = neg_outputs * tf.cast(~outputs_mask, dtype=tf.int64)
    outputs = tf.reduce_sum([outputs, neg_outputs], axis=0)

    return masked_inputs, outputs


def masked_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    """
    Compute masked loss.
    Args:
        y_true: True labels of input batch (shape: [batch_size, seq_length])
        y_pred: Predicted logits by model. (shape: [batch_size, seq_length, vocab_size])
    Returns:
        Masked (sparse) cross entropy loss (i.e. loss based only on masked tokens).
    """
    batch_size = tf.shape(y_true)[0]
    seq_len = tf.shape(y_true)[1]
    vocab_size = tf.shape(y_pred)[-1]
    mask = tf.cast(tf.math.not_equal(y_true, tf.constant(-100)), tf.float32)
    labels = y_true * tf.cast(mask, tf.int32)
    masked_proba = tf.nn.softmax(y_pred, axis=-1) * mask[..., tf.newaxis]
    dummy_proba = tf.concat([tf.constant([1.0]), tf.zeros(shape=vocab_size - 1)], axis=-1)
    dummy_proba = tf.tile([[dummy_proba]], multiples=[batch_size, seq_len, 1])
    dummy_proba = dummy_proba * (1 - mask)[..., tf.newaxis]
    final_proba = masked_proba + dummy_proba
    return cross_entropy_loss(labels, final_proba)


def masked_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor):
    """
    Masked accuracy metric.
    Args:
        y_true: True labels of input batch (shape: [batch_size, seq_length])
        y_pred: Predicted logits by model. (shape: [batch_size, seq_length, vocab_size])
    Returns:
        Masked accuracy.
    """
    mask = tf.cast(tf.math.not_equal(y_true, -100), tf.int32)
    y_true_masked = y_true * mask
    predictions = tf.argmax(y_pred, axis=-1)
    predictions_masked = tf.cast(predictions, tf.int32) * mask
    num_masked_tokens = tf.reduce_sum(tf.cast(tf.math.equal(y_true, -100), tf.int32))
    accuracy = tf.reduce_sum(tf.cast(tf.equal(y_true_masked, predictions_masked), tf.int32)) - num_masked_tokens
    return accuracy / tf.reduce_sum(mask)
