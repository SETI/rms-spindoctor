"""Keep a polyhedral-mesh body inside the keys the scene validator accepts.

The scene validator refuses every key in
:data:`spindoctor.sim.scene_schema.MESH_UNSUPPORTED_KEYS` on a
``shape_model: polyhedral_mesh`` body.  This module is the editor's side of that
rule, read from the same table so the two cannot drift:

- :data:`MESH_REFUSED_WIDGETS` names, for each refused key, the body-tab widget
  attributes whose controls write it.  Every key in the table must have an entry;
  a key without one raises ``KeyError`` the first time a body tab is built.
- :func:`sync_mesh_refused_widgets` disables those controls on a mesh body and
  gives each the validator's own sentence as its tooltip, then restores both on
  an ellipsoid.
- :class:`MeshKeyStash` holds the refused keys a body carried when it was
  switched to a mesh.  Switching moves them out of the scene, so a mesh body the
  editor holds or saves never carries a refused key, and switching back to an
  ellipsoid puts them back.
"""

from typing import Any

from PyQt6.QtWidgets import QWidget

from spindoctor.sim.scene_schema import MESH_UNSUPPORTED_KEYS, mesh_unsupported_message

# The body-tab widget attributes that write each refused key.  A key written by
# a checkable group maps to the group: disabling it disables every control
# inside, and leaves the children's own enable state (the Minnaert exponent's
# law gating) to come back when the group is re-enabled.
MESH_REFUSED_WIDGETS: dict[str, tuple[str, ...]] = {
    'atmosphere': ('atmosphere_group',),
    'photometric_law': ('photometry_group',),
    'minnaert_k': ('photometry_group',),
    'opposition_surge': ('surge_group',),
    'albedo_texture': ('albedo_group',),
    'disc_texture': ('disc_group',),
    'transits': ('transits_group',),
    'crater_fill': ('crater_fill_slider', 'crater_fill_spin'),
    'crater_min_radius': ('crater_min_radius_spin',),
    'crater_max_radius': ('crater_max_radius_spin',),
    'crater_power_law_exponent': ('crater_exponent_spin',),
    'crater_relief_scale': ('crater_relief_scale_spin',),
}

# Dynamic property holding a control's own (ellipsoid) tooltip while a mesh
# body shows the refusal in its place.
_OWN_TOOLTIP = 'ellipsoid_tooltip'


def _refused_keys_by_widget() -> dict[str, list[str]]:
    """Map each widget attribute to the refused keys it writes, in table order."""
    by_widget: dict[str, list[str]] = {}
    for key in MESH_UNSUPPORTED_KEYS:
        for attr in MESH_REFUSED_WIDGETS[key]:
            by_widget.setdefault(attr, []).append(key)
    return by_widget


def sync_mesh_refused_widgets(w: QWidget, is_mesh: bool) -> None:
    """Disable a mesh body's refused controls and show why, or restore them.

    On a mesh body every control in :data:`MESH_REFUSED_WIDGETS` is disabled and
    its tooltip becomes :func:`~spindoctor.sim.scene_schema.mesh_unsupported_message`
    for the keys it writes, the sentence the validator raises.  On an ellipsoid the
    controls are enabled and get their own tooltips back.

    Parameters:
        w: A built body tab carrying the attributes the table names.
        is_mesh: Whether the body is a ``polyhedral_mesh`` body.

    Raises:
        KeyError: If a key in ``MESH_UNSUPPORTED_KEYS`` has no table entry.
        AttributeError: If ``w`` lacks a widget the table names.
    """
    for attr, keys in _refused_keys_by_widget().items():
        widget: QWidget = getattr(w, attr)
        own = widget.property(_OWN_TOOLTIP)
        if own is None:
            own = widget.toolTip()
            widget.setProperty(_OWN_TOOLTIP, own)
        widget.setEnabled(not is_mesh)
        widget.setToolTip(mesh_unsupported_message(keys) if is_mesh else str(own))


class MeshKeyStash:
    """The refused keys each body carried when the editor switched it to a mesh.

    Entries are keyed by the body dict's identity and keep the dict itself, so an
    entry follows its body through tab rebuilds and index shifts, and a deleted
    body's id cannot be reused by another body while its entry lives.
    """

    def __init__(self) -> None:
        """Start with an empty stash."""
        self._entries: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}

    def stashed(self, body: dict[str, Any]) -> dict[str, Any]:
        """The refused keys stashed for ``body``, or an empty dict."""
        entry = self._entries.get(id(body))
        if entry is None or entry[0] is not body:
            return {}
        return entry[1]

    def take(self, body: dict[str, Any]) -> None:
        """Move every refused key ``body`` carries into the stash.

        Parameters:
            body: The scene's body dict, edited in place: afterwards it carries no
                key from ``MESH_UNSUPPORTED_KEYS``.
        """
        taken = {key: body.pop(key) for key in MESH_UNSUPPORTED_KEYS if key in body}
        if taken:
            self._entries[id(body)] = (body, {**self.stashed(body), **taken})

    def give_back(self, body: dict[str, Any]) -> None:
        """Return ``body``'s stashed keys to it and forget them.

        A key the body carries again keeps its current value.

        Parameters:
            body: The scene's body dict, edited in place.
        """
        for key, value in self.stashed(body).items():
            body.setdefault(key, value)
        self._entries.pop(id(body), None)

    def view(self, body: dict[str, Any]) -> dict[str, Any]:
        """``body`` with its stashed keys merged in, for building its controls.

        A tab rebuilt while the body is a mesh shows the values that switching back
        to an ellipsoid restores.  The returned dict is a new mapping; edits go to
        the scene's body dict.
        """
        return {**self.stashed(body), **body}
