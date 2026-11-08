"""Reviewed v1 geometry adapters; a rehashed release cannot change its scorer.

Water retains its exact parent sources. These new remote/storage adapters only
parameterize frozen physical geometry or public nominal masses. An additional
adapter version requires explicit new pins, preserving this version's values.
"""

REMOTE_SINGLE = {
    'sequence_evaluator.py': 'c1f556d6eedf789bc202d6a84b3f39ca6664499609dbfeac9ee9e4bdfc180077',
    'rules.py': '0d3acb43c7ac0dded40a1e0445d26b09f560c89079ce083702bd2168032d72b3',
}
REMOTE_DUAL = {
    'sequence_evaluator.py': '4d254ca869cd68f41c21d76dc438126e2dc890992873e1f2948052d04fa6a169',
    'rules.py': 'a516e63a6316b028a7f9244bfb1aefc5139c94ca82453e64c38b9a0037720227',
}
STORAGE = {
    'sequence_evaluator.py': 'e0e61c2fb2cc4f3a3bafb86d31581b1a1df1b92a82b2dc11fa95528179a021c7',
    'rules.py': '3287abe39830c743bcae23c86118c11badadaf2c965e89ac0662576622d6e91d',
}
