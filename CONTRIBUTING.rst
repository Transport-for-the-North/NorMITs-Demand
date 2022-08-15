===============
Ways Of Working
===============

.. role:: python(code)
   :language: python

.. _`Google Python Style Guide`: https://google.github.io/styleguide/pyguide.html
.. _`pydocstyle`: http://www.pydocstyle.org/en/stable/index.html
.. _`black`: https://github.com/psf/black
.. _`pylint`: https://github.com/PyCQA/pylint
.. _`mypy`: https://github.com/python/mypy
.. _`pyproject.toml`: pyproject.toml
.. _`todo comments`: https://google.github.io/styleguide/pyguide.html#312-todo-comments
.. _`old-style`: https://docs.python.org/3/library/stdtypes.html#old-string-formatting
.. _`new-style`: https://docs.python.org/3/library/stdtypes.html#str.format

GitHub
------

All new work should be worked on in a separate branch. For significant pieces of new functionality
a draft pull request should be opened during development of the code. This creates a central point
to communicate about the code being written and to keep a record of any potential issues. Once
development is complete and the code ready for review, the draft pull request can be promoted to
a normal pull request.

For smaller pieces of work, such as a bug fix, a normal pull request can be made straight away.

Coding Style
------------

NorMITs Demand follows Transport for the North's (TFN) coding standards, which include:

- Code must conform to `Google Python Style Guide`_
- Code uses numpy-style doc-strings, checked with `pydocstyle`_
- Code must be formatted with `black`_
- Code must be checked, and all errors corrected, by running `pylint`_
- Code must be checked, and all errors corrected, by running `mypy`_

See this project's `pyproject.toml`_ to see how these tools have been set up in order
to meet TFN's coding standards.

A few important cases are highlighted below:

- **Multiline for-statements** - Try to avoid these where possible to prevent
  visual conflicts. Instead, build the generator object before the
  :code:`for`-statement, then call the generator. Therefore, the following:

  .. code-block:: python

    for variable_one, variable_two, variable_three in zip(long_iterable_name_one,
                                                          long_iterable_name_two,
                                                          long_iterable_name_three):
        # Loop over one, two, and three
        do_something()

  Would become:

  .. code-block:: python

    one_two_three_iterator = zip(long_iterable_name_one,
                                 long_iterable_name_two,
                                 long_iterable_name_three)

    for variable_one, variable_two, variable_three in one_two_three_iterator:
        # Loop over one, two, and three
        do_something()

- **String Formatting** - Python has multiple different ways to format strings. Where
  possible avoid using concatenation to build strings as it can lead to difficult to
  read and error prone code. In order of preference, use either "`new-style`_"
  string formatting, or "`old-style`_" (similar to :code:`printf()` in C). For
  further information, see the links above.


Special Code Comments
---------------------
A few unique code comment prefixes have been chosen to help keep track of issues in the
codebase. They are noted below, with a brief description on when to use them:

- :code:`# TODO(BT):` Use this for small bits of future work, such as making a note of
  potential future errors that should be checked for, or code that could be better
  written if you had more time. A :code:`TODO` comment begins with the string :code:`TODO`
  in all caps and a parenthesized identifier (usually initials will do) of the
  person or issue with the best context about the problem. For further
  information, see `todo comments`_.

- :code:`# OPTIMISE:` Point out code that can be better optimised, but you don't
  have time/resources right now, i.e. re-writing code in numpy in place of Pandas.

- :code:`# BACKLOG:` Use to point out bigger pieces of work, such as where new
  (usually more complex) functionality can be added in future. Can also be used to
  point out where assumptions have been made in the codebase, and the backlog item
  can be used to track the issue.
