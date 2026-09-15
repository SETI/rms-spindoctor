"""How the bundle check reads the elements of a label.

A PDS4 label names its classes and attributes in the namespace of the dictionary that
defines them, so a child is looked for in its parent's namespace rather than in one this
package names.
"""

from typing import Any

from lxml import etree


def local_name(element: Any) -> str:
    """Return an element's name without its namespace.

    Parameters:
        element: An lxml element.

    Returns:
        The local part of its tag, as in ``Table_Character``.
    """
    return str(etree.QName(element).localname)


def child(element: Any, name: str) -> Any:
    """Return an element's first child of a name, in the element's own namespace.

    Parameters:
        element: An lxml element.
        name: The child's local name.

    Returns:
        The child, or None when the element has none of that name.
    """
    namespace = etree.QName(element).namespace
    tag = name if namespace is None else f'{{{namespace}}}{name}'
    return element.find(tag)


def children(element: Any, name: str) -> list[Any]:
    """Return every child of a name an element has, in the element's own namespace.

    Parameters:
        element: An lxml element.
        name: The children's local name.

    Returns:
        The children, in document order.
    """
    namespace = etree.QName(element).namespace
    tag = name if namespace is None else f'{{{namespace}}}{name}'
    return list(element.iterfind(tag))


def child_text(element: Any, *names: str) -> str | None:
    """Return the stripped text at a path of children below an element.

    Parameters:
        element: An lxml element.
        *names: The local names of the children to descend through, in order, each in
            the namespace of the element above it.

    Returns:
        The text of the last child, stripped of surrounding white space, or None when an
        element on the path is missing.
    """
    current = element
    for name in names:
        current = child(current, name)
        if current is None:
            return None
    return str(current.text or '').strip()


def child_integer(element: Any, *names: str) -> int | None:
    """Return the integer at a path of children below an element.

    Parameters:
        element: An lxml element.
        *names: The local names of the children to descend through, as
            :func:`child_text` takes them.

    Returns:
        The integer, or None when the path is missing or does not hold an integer, which
        the XML schema reports.
    """
    text = child_text(element, *names)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def element_path(element: Any) -> str:
    """Return where an element sits in its label, as a path of its ancestors' names.

    Each step is the element's name as the label writes it, its prefix included, and is
    numbered only when the element has siblings of the same name, so that the path
    names one element: ``/Product_Observational/File_Area_Observational/Header[2]``.

    Parameters:
        element: An lxml element.

    Returns:
        The path from the root element.
    """
    steps: list[str] = []
    current = element
    while current is not None:
        name = local_name(current)
        if current.prefix is not None:
            name = f'{current.prefix}:{name}'
        parent = current.getparent()
        if parent is not None:
            same = [sibling for sibling in parent if sibling.tag == current.tag]
            if len(same) > 1:
                name = f'{name}[{same.index(current) + 1}]'
        steps.append(name)
        current = parent
    return '/' + '/'.join(reversed(steps))
