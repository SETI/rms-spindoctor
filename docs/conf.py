#!/usr/bin/env python3

# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import functools
import importlib.metadata
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.ext.autodoc.mock import ismock

sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------

project = 'SpinDoctor'
copyright = '2025, SETI Institute'
author = 'SETI Institute'

# The full version, including alpha/beta/rc tags
try:
    release = importlib.metadata.version('rms-spindoctor')
except importlib.metadata.PackageNotFoundError:
    release = '1.0.0'  # fallback for development

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.mermaid',
    'myst_parser',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# The simulator image galleries carry a NOTES.md regeneration note alongside the
# committed PNG assets; exclude them so Sphinx does not treat them as orphan docs.
# The instrument chapter templates are what a new instrument's chapter is copied
# from, not documents in their own right, so they are excluded as well -- which
# also keeps them out of the glob toctree that discovers the chapters.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**/_sim_images/NOTES.md',
    '**/_scene_images/NOTES.md',
    '**/instruments/_template.rst',
]

# The suffix(es) of source filenames.
source_suffix = ['.rst', '.md']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Show every section level in the sidebar, with the branch holding the current
# page expanded and the others expandable in place. The Sidebar section below
# keeps the API reference's entries out of it outside the API reference.
html_theme_options = {
    'navigation_depth': -1,
    'collapse_navigation': False,
    'titles_only': False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

add_module_names = False
autodoc_typehints_format = 'short'
# Mock PyQt6 and matplotlib Qt backends so autodoc can import spindoctor.ui modules
# without a display or OpenGL context (e.g. in CI).
autodoc_mock_imports = [
    'PyQt6',
    'matplotlib.backends.backend_qtagg',
    'matplotlib.backends.backend_qt',
]

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx settings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'filecache': ('https://rms-filecache.readthedocs.io/en/latest/', None),
    'pdslogger': ('https://rms-pdslogger.readthedocs.io/en/latest/', None),
}

# Suppress nitpicky warnings for symbols that have no inventory we can link to:
# third-party packages without Sphinx docs (oops), test modules excluded from
# autodoc, sibling packages outside the importable spindoctor API surface, typing
# internals leaked by autodoc, and TypeVars / unqualified type aliases that
# Sphinx does not register as cross-reference targets.
nitpick_ignore_regex = [
    (r'py:.*', r'oops\..*'),
    (r'py:.*', r'tests\..*'),
    (r'py:.*', r'spindoctor\.cli\.backplanes\..*'),
    (r'py:.*', r'spindoctor\.cli\.ck(\..*)?'),
    (r'py:.*', r'spindoctor\.cli\.pds4\..*'),
    (r'py:.*', r'spindoctor\.cli\.reproj\..*'),
    (r'py:.*', r'numpy\._typing\..*'),
    (r'py:.*', r'argparse\._.*'),
    (r'py:.*', r'spindoctor\.support\.types\.NPType'),
    (r'py:.*', r'spindoctor\.ui\.mosaic_viewer\..*'),
]

# MyST-Parser settings
myst_enable_extensions = [
    'colon_fence',
    'deflist',
]

# Mermaid settings: use default client-side rendering (no mmdc/Chromium) so docs
# build in headless CI; diagrams render in the browser via mermaid.js.
mermaid_d3_zoom = True

# -- Sidebar -----------------------------------------------------------------

# The sidebar lists every guide and report page, at full depth, on every page.
# The API reference appears in it as one entry with no children, except on the
# API reference's own pages, where its pages and their documented objects are
# listed as well. The API reference holds thousands of entries, one per
# documented class, function, and method, and writing them all into every page
# made each page about 1 MB and the build several times slower.
#
# sphinx_rtd_theme draws the sidebar by calling the page context's ``toctree``
# function. Outside the API reference, _sidebar_toctree replaces that function
# with one that resolves the same global toctree while the API reference's
# table of contents is swapped for a copy holding only its title, so Sphinx
# never copies, resolves, or renders the API entries for that page. On an API
# reference page the theme's own call is left in place.

API_REFERENCE_DOC = 'api_reference'


def _is_api_reference_page(pagename: str) -> bool:
    """Return whether a page belongs to the API reference."""
    return pagename == API_REFERENCE_DOC or pagename.startswith(f'{API_REFERENCE_DOC}/')


def _title_only_toc(toc: nodes.Element) -> nodes.Element:
    """Return a copy of a table of contents without its toctrees or empty lists."""
    toc = toc.deepcopy()
    for toctree_node in list(toc.findall(addnodes.toctree)):
        toctree_node.parent.remove(toctree_node)
    for bullet_list in reversed(list(toc.findall(nodes.bullet_list))):
        if not bullet_list.children and bullet_list.parent is not None:
            bullet_list.parent.remove(bullet_list)
    return toc


def _sidebar_toctree(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: dict[str, Any],
    doctree: nodes.document | None,
) -> None:
    """Make the sidebar of a page outside the API reference omit the API entries."""
    if _is_api_reference_page(pagename) or 'toctree' not in context:
        return
    builder = app.builder

    def toctree(**kwargs: Any) -> str:
        tocs = builder.env.tocs
        full_toc = tocs[API_REFERENCE_DOC]
        tocs[API_REFERENCE_DOC] = _title_only_toc(full_toc)
        try:
            # The same call the HTML builder's own toctree function makes.
            return builder._get_local_toctree(pagename, **kwargs)
        finally:
            tocs[API_REFERENCE_DOC] = full_toc

    context['toctree'] = toctree


# -- Source links ------------------------------------------------------------

# Each documented object links to its lines on GitHub at the commit being
# built: Read the Docs supplies that commit, a local build asks git, and a
# checkout git cannot read links to main. An object whose source cannot be
# located exactly (a value, an attribute holding an object defined under
# another name, or anything from a mocked module) gets no link.

GITHUB_BLOB_URL = 'https://github.com/SETI/rms-spindoctor/blob'


def _source_ref() -> str:
    """Return the commit the links point at."""
    commit = os.environ.get('READTHEDOCS_GIT_COMMIT_HASH')
    if commit:
        return commit
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return 'main'
    return result.stdout.strip() or 'main'


SOURCE_REF = _source_ref()


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Return the GitHub URL of a documented object's source lines, or None."""
    if domain != 'py':
        return None
    module = sys.modules.get(info.get('module', ''))
    package = sys.modules.get('spindoctor')
    if module is None or package is None or package.__file__ is None:
        return None
    obj: Any = module
    for part in info['fullname'].split('.'):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
    if ismock(obj):
        return None
    if isinstance(obj, property):
        obj = obj.fget
    elif isinstance(obj, functools.cached_property):
        obj = obj.func
    try:
        obj = inspect.unwrap(obj)
    except ValueError:
        return None
    # A value, or an attribute that holds something defined under another name.
    if getattr(obj, '__name__', None) != info['fullname'].split('.')[-1]:
        return None
    try:
        source_file = inspect.getsourcefile(obj)
        lines, start = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        return None
    if source_file is None:
        return None
    try:
        path = Path(source_file).resolve().relative_to(Path(package.__file__).resolve().parent)
    except ValueError:
        return None
    end = start + len(lines) - 1
    return f'{GITHUB_BLOB_URL}/{SOURCE_REF}/src/spindoctor/{path.as_posix()}#L{start}-L{end}'


def setup(app: Sphinx) -> None:
    """Connect the sidebar handler."""
    app.connect('html-page-context', _sidebar_toctree)
