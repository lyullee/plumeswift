"""Input-relative path resolution with the QES filesystem rules."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QESInputPaths:
    directory: Path

    @classmethod
    def from_input(cls, filename):
        # Canonicalize the supplied input's parent, not a symlink target's
        # parent. This also avoids shared mutable state between XML readers.
        return cls(Path(filename).parent.resolve(strict=True))

    def resolve(self, filename, *, require_parent=True):
        path = Path(filename)
        if path.is_absolute():
            return path
        path = self.directory/path
        return path.parent.resolve(strict=require_parent)/path.name
