.. _installation:

Installation
============

``calistar`` is available on `PyPI <https://pypi.org/project/calistar/>`_ and `Github <https://github.com/tomasstolker/calistar>`_.

Installation from PyPI
----------------------

The ``calistar`` tool can be installed with the `pip package manager <https://packaging.python.org/tutorials/installing-packages/>`_:

.. code-block:: console

    $ pip install calistar

Or, to update ``calistar`` to the most recent version:

.. code-block:: console

   $ pip install --upgrade calistar

Installation from Github
------------------------

Using pip
^^^^^^^^^

The repository on `Github <https://github.com/tomasstolker/calistar>`_ contains the latest implementations and can also be installed with `pip <https://packaging.python.org/tutorials/installing-packages/>`_:

.. code-block:: console

    $ pip install git+https://github.com/tomasstolker/calistar.git

Cloning the repository
^^^^^^^^^^^^^^^^^^^^^^

In case you want to look into the code, it is best to clone the repository:

.. code-block:: console

    $ git clone https://github.com/tomasstolker/calistar.git

Next, the package is installed by running ``pip`` in the local repository folder:

.. code-block:: console

    $ pip install -e .

New commits can be pulled from Github once a local copy of the repository exists:

.. code-block:: console

    $ git pull origin main

Do you want to make changes to the code? Please fork the `calistar` repository on the Github page and clone your own fork instead of the main repository. Contributions and pull requests are welcome (see :ref:`about` section).

Testing `calistar`
------------------

The installation can now be tested, for example by starting Python in interactive mode and importing the package:

.. code-block:: python

    >>> import calistar
