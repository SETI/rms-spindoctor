from typing import Any, no_type_check


@no_type_check
class AttrDict(dict[str, Any]):
    """Implements a dictionary that allows attribute-style access to its key-value pairs.

    A dictionary subclass that exposes its keys as attributes, allowing dict items to be
    accessed using attribute notation (dict.key) in addition to the normal dictionary
    lookup (dict[key]).

    Because the instance dict and the mapping are the same object, setting an attribute
    inserts a key.  The class therefore carries the marker that keeps oops from writing
    its mutability bookkeeping into one; see the comment on it below.

    Parameters:
        *args: Variable length argument list passed to dict constructor.
        **kwargs: Arbitrary keyword arguments passed to dict constructor.
    """

    # ``_IS_IMMUTABLE`` is oops's own marker, read by ``oops.mutable._get_info``,
    # and this is the only reason it appears here.  That function caches its
    # mutability verdict by setting an attribute on every object it walks, and
    # it walks everything reachable from an Observation.  Building a Backplane
    # therefore reaches the shared Config through the observation and so every
    # section of it; because this class publishes its own dict as its
    # ``__dict__``, the attribute oops sets becomes a configuration key that no
    # configuration file declares.  A section read back, copied, or compared
    # would then carry a key its file never wrote.
    # The marker lives on the class, so oops finds it with ``hasattr`` without
    # it occupying the mapping.
    #
    # It is an opt-out from one package rather than a general convention: it
    # works only because oops honors it, and it depends on a private upstream
    # name.  Remove it once oops keeps its bookkeeping on its own objects.
    # See #552.
    _IS_IMMUTABLE = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    # This is a stupid thing to do, but it's necessary to avoid mypy from complaining
    # about missing attributes. mypy ignores attributes for classes that have a
    # __getattr__ method.
    @no_type_check
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"Attribute '{name}' not found") from exc
