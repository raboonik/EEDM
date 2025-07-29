'''
    Shared memory context
'''

import settings as set

comm     = None
rank     = None
size     = None
mainrank = 0


eq6Cond  = True
eq9Cond  = True

# Auxiliary booleans for neater syntaxes
EnCond   = set.savePredecompE
SpCond   = set.saveSpeeds
ErCond   = set.saveDivB
DbCond   = set.saveDivBErr
PkCond   = set.savePolytropK