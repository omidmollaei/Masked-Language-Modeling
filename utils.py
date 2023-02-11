"""
Parse CLI arguments based on Model dataclass and Datasets dataclass
"""

import dataclasses

from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter

from typing import Any, Iterable, Dict, NewType, get_type_hints

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
            print(field.name)
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
