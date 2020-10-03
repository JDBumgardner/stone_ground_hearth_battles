from typing import List


class FrozenList:
    def __init__(self, l: List):
        self._private_list_dont_modify = l

    def copy(self):
        """ Return a shallow copy of the list. """
        return self._private_list_dont_modify.copy()

    def count(self, x):
        """ Return number of occurrences of value. """
        return self._private_list_dont_modify.count(x)

    def index(self, x, start=0, stop=9223372036854775807):
        """
        Return first index of value.

        Raises ValueError if the value is not present.
        """
        return self._private_list_dont_modify.index(x, start, stop)

    def __contains__(self, x):
        """ Return key in self. """
        return x in self._private_list_dont_modify

    def __eq__(self, value):
        """ Return self==value. """
        return self._private_list_dont_modify == value

    def __getitem__(self, key):
        if isinstance(key, slice):
            return FrozenList(self._private_list_dont_modify[key])
        return self._private_list_dont_modify[key]

    def __ge__(self, value):
        return self._private_list_dont_modify >= value

    def __gt__(self, value):
        """ Return self>value. """
        return self._private_list_dont_modify > value

    def __iter__(self):
        """ Implement iter(self). """
        return iter(self._private_list_dont_modify)

    def __len__(self):
        """ Return len(self). """
        return len(self._private_list_dont_modify)

    def __le__(self, value):
        """ Return self<=value. """
        return self._private_list_dont_modify <= value

    def __lt__(self, value):
        """ Return self<value. """
        return self._private_list_dont_modify < value

    def __mul__(self, value):
        """ Return self*value. """
        return self._private_list_dont_modify * value

    def __ne__(self, value):
        """ Return self!=value. """
        return self._private_list_dont_modify != value

    def __repr__(self):
        """ Return repr(self). """
        return repr(self._private_list_dont_modify)

    def __reversed__(self):
        """ Return a reverse iterator over the list. """
        return reversed(self._private_list_dont_modify)

    def __rmul__(self, value):  # real signature unknown
        """ Return value*self. """
        return value * self._private_list_dont_modify

    def __add__(self, other):
        return self._private_list_dont_modify + other

    def __radd__(self, other):
        return other + self._private_list_dont_modify

    __hash__ = None
