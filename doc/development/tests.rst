.. module:: ase.test

================
Testing the code
================

All additions and modifications to ASE should be tested.

.. index:: testase

Tests should be put in the :git:`ase/test` directory.
Run all tests with::

  ase test

This requires installing pytest and pytest-xdist.
See ``ase test --help`` for more information.

You can also run ``pytest`` directly from within the ``ase/test`` directory.

.. important::

  When you fix a bug, add a test to the test suite checking that it is
  truly fixed.  Bugs sometimes come back, do not give it a second
  chance!


How to add a test
=================

Create a module somewhere under ``ase/test``.  Make sure its name
starts with ``test_``.  Inside the module, each test should be a
function whose name starts with ``test_``.  This ensures that pytest
finds the test.  Use ``ase test --list`` to see which tests it will
find.

You may note that many tests do not follow these rules.
These are older tests.  We expect to port them one day.


How to write good tests
=======================

Clearly written tests function both as a specification of the code's
behavior and documentation of the code's interface.  If the tests are
complete enough, we can safely change the code and know that it still
works because the tests pass.

In order to *know* that the tests, and hence that they are complete enough
that we can rely on them as an indicator of whether the code actually works,
they should be grouped logically, be readable, and so on.

 * Think of tests as having three steps: Setup, execution, and assertion.

 * Write many small tests, not few large tests.  Good tests are often
   only 3-5 lines long.

 * Use pytest fixtures to handle the setup -- and if necessary teardown --
   of objects required by tests.

 * Try to make each test contain at least one assertion.
   Tests without assertion are "toothless" as the code can
   malfunction in many ways and we'll never know.

Admonitions:

 * Avoid duplication in tests.  Use pytest fixtures instead.
   The fixture can return a single object, or if you need many
   differently configured object, write a *factory* fixture which
   returns a function which produces objects.

 * Don't call a "big long" function to test only one tiny aspect of
   its behaviour.  This becomes a temptation whenever a function does
   too many things.  Instead, split up the function into smaller
   functions, where each function does only one thing.  Then test
   those functions.

 * Don't overdo it.  We shouldn't have too *many* tests cover the same
   production code -- that makes it difficult to change the code.

 * If a class is difficult to test because it can change its state in
   different ways, and we need to test many different code paths to
   exercise each way the state can change, then that class is probably
   too complicated and needs to be rewritten.


The old test suite
==================

In the old test suite, each test was a separate file, and running the test
meant running that file.  That's
conceptually simple, but encourages bad design: Tests tend to be long
and "rambling".
Many older tests read like a long-winded adventure,
going through an intricate plot with twists and surprises, each step
depending on the former.  Such a tale makes exciting literature,
but as tests they are terrible.

Good tests are the exact opposite: A good test tests one thing and
only that thing.  It is named after what it tests.
To test multiple things, there are multiple independent tests, or one
test parametrized over multiple inputs.


How to test I/O formats
=======================

Something about splitting up into small functions and testing
those functions individually.

How to test calculators
=======================

Something about dockers

Something about calculator factories

How to fail successfully
========================

The test suite provided by :mod:`ase.test` automatically runs all test
scripts in the :git:`ase/test` directory and summarizes the results.

If a test script causes an exception to be thrown, or otherwise terminates
in an unexpected way, it will show up in this summary. This is the most
effective way of raising awareness about emerging conflicts and bugs during
the development cycle of the latest revision.


Remember, great tests should serve a dual purpose:

**Working interface**
    To ensure that the :term:`class`'es and :term:`method`'s in ASE are
    functional and provide the expected interface. Empirically speaking, code
    which is not covered by a test script tends to stop working over time.

**Replicable results**
    Even if a calculation makes it to the end without crashing, you can never
    be too sure that the numerical results are consistent. Don't just assume
    they are, :func:`assert` it!

.. function:: assert(expression)

    Raises an ``AssertionError`` if the ``expression`` does not
    evaluate to ``True``.



Example::

  from ase import molecule

  def test_c60():
      atoms = molecule('C60')
      atoms.center(vacuum=4.0)
      result = atoms.get_positions().mean(axis=0)
      expected = 0.5*atoms.get_cell().diagonal()
      tolerance = 1e-4
      assert (abs(result - expected) < tolerance).all()


To run the same test with different inputs, use pytest fixtures.
For example::

  @pytest.mark.parametrize('parameter', [0.1, 0.3, 0.7])
  def test_something(parameter):
      # setup atoms here...
      atoms.set_something(parameter)
      # calculations here...
      assert everything_is_going_to_be_alright
