===============
Ways Of Working
===============

.. role:: python(code)
   :language: python

.. _`Google Python Style Guide`: https://google.github.io/styleguide/pyguide.html

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

NorMITs Demand follows the `Google Python Style Guide`_ with some exceptions:

- **Multiline :code:`if`-statements** - To avoid a visual conflict with multiline
  :code:`if`-statements this codebase chooses to forgo the space between :code:`if`
  and the opening bracket. A comment should also be added, if reasonable. Therefore,
  the following:

  .. code-block:: python

    if (this_is_one_thing and
        this_is_another_thing):
        do_something()


  Would become:

  .. code-block:: python

    if(this_is_one_thing and
       this_is_another_thing):
       # As both conditions are true, we should do the thing
        do_something()

- **Multiline :code:`for`-statements** - Try to avoid these where possible to prevent
  (less obstructive) visual conflicts. Instead, build the generator object before the
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

- **Function definitions** - To clearly differentiate function definitions from function
  calls at a glance, all function arguments should be on a separate line, in-line with
  the opening bracket. The trailing bracket should also be in line with the arguments,
  on a new line (along with return type hinting).
  Note: If all function arguments can fit on a single line with the function name, then
  that is the preferred option. As an example:

  .. code-block:: python

    def a_long_function_name(variable_name_one: int,
                             variable_name_two: int,
                             variable_name_three: int,
                             variable_name_four: int,
                             ) -> float:
        # Do something
    ...

Special Code Comments
---------------------
A few unique code comment prefixes have been chosen to help keep track of issues in the
codebase. They are noted below, with a brief description on when to use them:

- :code:`# TODO:` Use this for small bits of future work, such as making a note of
  potential future errors that should be checked for, or code that could be better
  written if you had more time.

- :code:`# OPTIMISE:` Point out code that can be better optimised, but you don't
  have time/resources right now, i.e. re-writing code in numpy in place of Pandas.

- :code:`# BACKLOG:` Use to point out bigger pieces of work, such as where new
  (usually more complex) functionality can be added in future. Can also be used to
  point out where assumptions have been made in the codebase, and the backlog item
  can be used to track the issue.
