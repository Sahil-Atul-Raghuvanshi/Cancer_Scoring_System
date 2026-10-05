"""Turning a scored case into a file somebody outside this system can read.

Nothing here computes anything. Every number a workbook prints comes from a
report that is already on disk, so the spreadsheet and the API cannot disagree
about what was measured - and a disputed cell can be traced to the JSON it was
read from.
"""
