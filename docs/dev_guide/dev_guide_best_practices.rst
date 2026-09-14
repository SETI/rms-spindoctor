==============
Best Practices
==============

Project conventions for all new and modified code live in
``.cursor/rules/python.mdc`` (with peer files for
``dependency_management``, ``doc_python``, ``environment``,
``git_workflow``, ``pull_request``, ``python_testing``, and
``security``). That file is the
authoritative standard; this page lists the rules that come up most often.

Code style
----------

* Maximum line length: 100 characters (enforced by Ruff).
* Naming: ``lowercase_with_underscores`` for functions and local variables,
  ``TitleCase`` for classes, ``ALL_CAPS_WITH_UNDERSCORES`` for module-level
  constants. Prefix non-public names with a single underscore.
* Type-annotate every function parameter and return value, including
  ``-> None``. Use modern ``list[str]``, ``dict[str, int]``, ``X | None``
  syntax.
* Keep modules under 1000 lines. Split larger modules into a package.
* Do not introduce compatibility shims for prior versions unless explicitly
  requested; change the code instead.

Imports
-------

* Imports go at the top of the file, in three alphabetically sorted groups
  separated by blank lines: standard library, third party, then this project.
* An import inside a function is permitted only to keep a heavy dependency off
  a path that does not use it, and it must carry a comment naming the
  dependency and the path. Most of them keep an optional or costly library out
  of a module that only some callers reach: the GUI toolkit, the plotting and
  imaging libraries, the SPICE and array libraries a single function needs, and
  each instrument's ``oops`` host module.
* Two of them protect a guarantee beyond their own module:

  * PyQt6, imported where a dialog is opened rather than at the top of the
    module that opens it -- the manual navigation technique in
    ``spindoctor/nav_technique/nav_technique_manual.py`` is the one on the
    navigation path -- so that a headless run never imports a GUI toolkit.
  * :func:`spindoctor.results_index.masked_url` in
    ``spindoctor/support/command_line.py``, imported only when the command line
    being logged carries a value for a connection-URL option. Whether such a
    value holds a credential is what the masking rule itself decides, so
    locating one is as far as the module gets without it. The run banner of
    every program passes through that module and most runs name no index, so a
    top-level import would put the database layer behind every one of them.
    A subprocess test in ``tests/spindoctor/support/test_command_line.py``
    asserts that masking a command line carrying no URL imports no
    :mod:`sqlalchemy` module.

Linting and typing
------------------

* Run ``ruff check src tests`` and ``ruff format --check src tests`` on the
  full codebase after changes.
* Run ``mypy src tests`` and fix every error. Do not add module-level
  ``# mypy: ignore-errors`` or global ``exclude`` entries. A line-level
  ``# type: ignore[error-code]`` is acceptable only with a brief
  justification.

Testing
-------

* Use ``pytest`` with ``pytest-xdist``; the canonical command is
  ``pytest -n 4 --dist=loadfile`` (the ``--dist=loadfile`` flag is
  required because PyQt6 workers crash under default xdist scheduling).
  Use ``-n 4``, never ``-n auto``, and pin the BLAS / OpenMP thread counts
  before running::

      export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

* Annotate test function parameters and return types; return ``-> None``.
* One assertion per condition (no ``and`` in assertions). When testing
  exceptions, use ``pytest.raises`` as a context manager and assert on the
  exception message content via ``match=``.
* Target at least 90 % line coverage over the full suite.

Documentation
-------------

* Every module, class, function, and method has a docstring written in
  Google style with ``Parameters:``, ``Returns:``, and ``Raises:`` as
  needed. Wrap docstring text to 90 characters.
* Do not use smart quotes, em-dashes, or arrows in ``.py`` files (they are
  fine in ``.rst`` and ``.md``).
* Update docstrings when the associated code changes, and remove them when
  the code is removed.
* After a code change, run ``sphinx-build -W -b html docs docs/_build`` and
  fix every warning.
