"""Typed value and reference exchange for QES component factories."""
from copy import deepcopy


class QESDataTransport:
    """Python counterpart of the native name-to-std::any container.

    Value entries are copied on insertion and on value retrieval. Explicit
    reference entries share the original object, like a native pointer entry.
    """

    def __init__(self):
        self._entries = {}

    def put(self, key, value, *, reference=False):
        self._entries[key] = (value if reference else deepcopy(value), reference)

    def _entry(self, key, expected_type):
        if key not in self._entries:
            raise RuntimeError(f'Key not found: {key}')
        value, reference = self._entries[key]
        if type(value) is not expected_type:
            raise RuntimeError(f'Type mismatch for key: {key}')
        return value, reference

    def get(self, key, expected_type):
        value, reference = self._entry(key, expected_type)
        return value if reference else deepcopy(value)

    def get_ref(self, key, expected_type):
        return self._entry(key, expected_type)[0]

    def contains(self, key):
        return key in self._entries

    def clear(self):
        self._entries.clear()
