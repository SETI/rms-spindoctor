======================
Installation and Setup
======================

See :doc:`/introduction_overview` for package installation with ``pip`` or
``pipx``.

Environment Setup
-----------------

In addition to installing the package, the following external resources are
needed at runtime.

**SPICE kernels.**  Download the SPICE kernels required for your mission and
set ``SPICE_PATH`` to the directory that contains them:

.. code-block:: bash

   export SPICE_PATH=/path/to/your/spice/kernels

**PDS3 holdings.**  For PDS3 datasets (all currently supported missions), set
``PDS3_HOLDINGS_DIR`` to the root of a PDS3 holdings tree (or pass
``--pds3-holdings-root`` on the command line):

.. code-block:: bash

   export PDS3_HOLDINGS_DIR=/path/to/your/pds3/data

The holdings tree follows the layout used by the PDS Ring-Moon Systems Node::

   $PDS3_HOLDINGS_DIR/
       volumes/
           <volume_set>/
               <volume>/
                   <data directories>/
       metadata/
           <volume_set>/
               <volume>/
                   <volume>_index.lbl
                   <volume>_index.tab

Remote holdings are supported: ``PDS3_HOLDINGS_DIR`` and
``--pds3-holdings-root`` accept any URL understood by ``filecache.FCPath``
(for example ``https://pds-rings.seti.org/holdings``).


