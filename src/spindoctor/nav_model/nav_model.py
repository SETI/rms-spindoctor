"""NavModel — the base class for predicted-scene generators.

Each NavModel renders the predicted appearance of one part of the scene
(stars, a body, or rings) given the observation's SPICE prediction.  The
orchestrator iterates registered NavModel instances and asks each to
contribute features and annotations to the navigation pipeline:

- ``create_model`` populates the model's internal state and metadata.
- ``to_features(context)`` returns ``NavFeature`` instances ready for
  technique consumption.
- ``to_annotations(context)`` returns an ``Annotations`` collection for
  the summary PNG.

Concrete NavModel subclasses self-register via ``__init_subclass__``, and each
declares in ``inputs`` the model source it builds from.  The module-level
``build_models_for_obs`` function iterates the registry and asks each subclass
whose ``inputs`` matches the observation's ``model_source`` to construct whatever
instances apply to the observation; this is what the top-level navigation driver
uses so it does not need to know which scene categories exist.

Every NavModel inherits from ``NavBase`` so it gets the project logger and
config plumbing automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from spindoctor.annotation import Annotations
from spindoctor.config import Config
from spindoctor.feature.feature import NavFeature
from spindoctor.obs import ObsSnapshotInst
from spindoctor.obs.obs_inst import ModelSource
from spindoctor.support.nav_base import NavBase

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['NavModel', 'build_models_for_obs']


class NavModel(NavBase, ABC):
    """Base class for predicted-scene generators.

    Class attributes:
        inputs: The model source the class builds its predictions from.  Every
            concrete subclass sets it in its own class body; the builder asks the
            class for instances only when it equals the observation's
            ``model_source``.
        _registry: List of every concrete subclass discovered at import
            time.  Iterated by ``build_models_for_obs`` to construct the
            per-image NavModel set.

    Parameters:
        name: The model's name (e.g. ``'stars'``, ``'body:MIMAS'``,
            ``'rings:SATURN'``).  Used as the lookup key in the
            orchestrator's ``model_metadata`` dict.
        obs: Observation snapshot the model is built against.
        config: Optional ``Config`` override (defaults to ``DEFAULT_CONFIG``).
    """

    inputs: ClassVar[ModelSource]
    _registry: ClassVar[list[type[NavModel]]] = []
    _abstract: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register concrete subclasses.

        Subclasses that exist only as shared bases (annotation helpers,
        for example) opt out by setting ``_abstract: ClassVar[bool] =
        True`` in the class body, and so do test fakes that are constructed
        directly and must stay out of the registry.

        Raises:
            TypeError: If a concrete subclass does not set ``inputs`` in its own
                class body.  A value inherited from a concrete parent does not
                count, so a simulated sibling of a production model cannot
                silently take the parent's model source.
        """
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get('_abstract', False):
            return
        if 'inputs' not in cls.__dict__:
            raise TypeError(
                f'{cls.__qualname__} must set inputs to the model source it builds from'
            )
        NavModel._registry.append(cls)

    def __init__(self, name: str, obs: ObsSnapshotInst, *, config: Config | None = None) -> None:
        super().__init__(config=config)
        self._name = name
        self._obs = obs
        self._metadata: dict[str, Any] = {}

    @property
    def name(self) -> str:
        """The model's name."""
        return self._name

    @property
    def obs(self) -> ObsSnapshotInst:
        """The observation snapshot this model is built against."""
        return self._obs

    @property
    def metadata(self) -> dict[str, Any]:
        """Per-model diagnostic dict; populated during ``create_model``."""
        return self._metadata

    @classmethod
    def instances_for_obs(
        cls, obs: ObsSnapshotInst, *, config: Config | None = None
    ) -> list[NavModel]:
        """Return concrete NavModel instances applicable to ``obs``.

        Default returns ``[]``.  Subclasses that auto-instantiate from an
        observation (stars, body, rings) override this to construct the
        appropriate set of instances — typically one per body in FOV, one
        per planet with visible rings, one stars model.

        ``build_models_for_obs`` calls this only for an observation whose
        ``model_source`` equals the class's ``inputs``, so an override need
        not check what kind of observation it was given.

        Parameters:
            obs: Observation snapshot to inspect.
            config: Configuration used both to decide which instances apply
                and to construct them.  None uses ``DEFAULT_CONFIG``.

        Returns:
            Zero or more NavModel instances.
        """
        del obs, config
        return []

    @abstractmethod
    def create_model(self) -> None:
        """Populate the model's internal state and ``metadata``.

        Called once per image before ``to_features`` or ``to_annotations``.
        Implementations evaluate the SPICE prediction, render any required
        templates, and populate ``self._metadata`` with diagnostic
        information.
        """

    @abstractmethod
    def to_features(self, context: NavContext) -> list[NavFeature]:
        """Return navigation-ready features for this model.

        Called after ``create_model``.  Implementations build per-element
        ``NavFeature`` instances (stars, body limbs, ring edges) and assign
        each its preferred filter, position covariance, and reliability.

        Parameters:
            context: Per-image ``NavContext`` carrying global statistics.

        Returns:
            List of ``NavFeature`` instances; may be empty if no usable
            features are present in this scene.
        """

    @abstractmethod
    def to_annotations(self, context: NavContext) -> Annotations:
        """Return a collection of annotations describing the predicted scene.

        Called after ``create_model``.  Annotations summarize what the
        model predicts is visible in the image; they end up on the summary
        PNG regardless of whether the features they describe end up gated
        out or unused by techniques.

        Parameters:
            context: Per-image ``NavContext``.

        Returns:
            ``Annotations`` collection (possibly empty).
        """


def build_models_for_obs(obs: ObsSnapshotInst, *, config: Config | None = None) -> list[NavModel]:
    """Construct every NavModel applicable to ``obs``.

    Iterates ``NavModel._registry`` and lets each registered concrete
    subclass whose ``inputs`` equals ``obs.model_source`` build whatever
    instances apply; a class that builds from another model source is never
    asked.  That one comparison is the whole choice between the models that
    predict a frame from SPICE geometry and the ones that read a simulated
    scene.  The top-level navigation driver uses this so the choice of which
    scene categories are represented (stars, body, rings, ...) lives entirely
    inside the NavModel subclasses, not in the driver.

    Parameters:
        obs: Observation snapshot.
        config: Configuration used both to decide which instances apply and
            to construct them, so a per-run override changes model selection
            the same way it changes model behavior.  None uses
            ``DEFAULT_CONFIG``.

    Returns:
        Flat list of NavModel instances ready for the orchestrator.
    """
    out: list[NavModel] = []
    for cls in NavModel._registry:
        if cls.inputs == obs.model_source:
            out.extend(cls.instances_for_obs(obs, config=config))
    return out
