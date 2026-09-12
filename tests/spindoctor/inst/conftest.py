"""Shared stand-ins for the instrument host tests."""

from typing import Any


class VicarLabelStandIn(dict[str, Any]):
    """A VICAR label's items, read the way rms-vicar's label reads them.

    ``VicarLabel.get`` takes its default as a required argument, so a host reading an item
    a label may lack has to pass one.  This stand-in refuses a call without it the same
    way, which a plain dict would not.
    """

    def get(self, key: str, default: Any) -> Any:  # type: ignore[override]  # VicarLabel.get requires the default
        """Return an item, or the default when the label lacks it.

        Parameters:
            key: The item's name.
            default: What to return when the label lacks the item.

        Returns:
            The item's value, or the default.
        """
        return super().get(key, default)
