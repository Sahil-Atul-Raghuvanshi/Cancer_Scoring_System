"""Maths and image primitives more than one pipeline step needs.

A step package owns its own algorithm. What lands here is the arithmetic two
steps would otherwise each keep a private copy of - and a private copy is how
two steps quietly stop agreeing about what a number means.
"""
