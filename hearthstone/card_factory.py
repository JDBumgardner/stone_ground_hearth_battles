from typing import Callable, Any, Optional, Tuple, Type


def make_metaclass(metaclass_callback: Callable[[Type], None], exclude: Optional[Tuple[str, ...]] = None) -> Type:
    """
    make a metaclass

    :param metaclass_callback: called when a new class is made using the metaclass
    :param exclude: names of inheriting classes to not trigger the callback on
    :return: the metaclass
    """
    exclude = exclude or ()

    class Metaclass(type):
        def __new__(mcs, name, bases, kwargs):
            klass = super().__new__(mcs, name, bases, kwargs)
            if name not in exclude:
                metaclass_callback(klass)
            return klass

    return Metaclass
