"""Write the fixture tree again, as the regeneration entry point.

Running the package writes every document through the writer and prints the
paths it wrote, which is the step a change to the writer or to a document is
re-ratified by.
"""

from . import RESULTS_TREE, write_results_tree

for written_path in write_results_tree(RESULTS_TREE):
    print(written_path)
