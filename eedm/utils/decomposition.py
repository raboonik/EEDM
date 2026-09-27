'''
    The output groupings of the eigenenergies (settings.decomposition). Each Paper III mode m = 1, ..., 9
    (Table 1) along each direction q contributes to exactly one output:

    (the direction letters are the simulation's axes; the gravity mode m9 acts along the vertical, settings.verticalAxis)
    "MHD_modes_xyz"    : every mode m along every direction q         -> m1_x, ..., m8_z, m9_z
    "MHD_branches_xyz" : reverse (-) + forward (+) of each branch, per q -> div_q, ent_q, A_q, s_q, f_q, g_z
    "MHD_net_branches" : each branch summed over -/+ and over x, y, z  -> div, ent, A, s, f, g
'''

from .. import context as ct
from .orientation import sim_axis

# The branches of Paper III and their modes: A = Alfven (m3 reverse, m4 forward), s = slow (m5, m6), f = fast (m7, m8).
# The gravity mode m9 only exists along the EEDM z axis, i.e. the simulation's vertical (settings.verticalAxis).
branches = (('div', (1,)), ('ent', (2,)), ('A', (3, 4)), ('s', (5, 6)), ('f', (7, 8)), ('g', (9,)))


def outputs(gravity=True):
    '''
        Ordered {output name: [(m, q), ...]} for the chosen decomposition; the g output is left out if gravity is False.
        The directions q are those of the EEDM frame (gravity along -z), in which Equation 6 is computed, while the output
        names use the simulation's own axes (utils.orientation), e.g. m9_y when settings.verticalAxis = "y".
    '''
    out = {}
    for b, modes in branches:
        if b == 'g' and not gravity: continue
        dirs = sorted('z' if b == 'g' else 'xyz', key=sim_axis)      # EEDM directions, in the order of their simulation axes
        if ct.decomposition == "MHD_modes_xyz":
            for m in modes:
                for q in dirs: out['m%d_%s' % (m, sim_axis(q))] = [(m, q)]
        elif ct.decomposition == "MHD_branches_xyz":
            for q in dirs: out['%s_%s' % (b, sim_axis(q))] = [(m, q) for m in modes]
        else:
            out[b] = [(m, q) for m in modes for q in dirs]
    return out


def placeholders(gravity=True):
    '''The outputs that are identically zero because all their modes lie along invariant axes (settings.invariantAxes);
    they are stored as [0] placeholders, like eq6_m9_x and eq6_m9_y.'''
    return [name for name, mqs in outputs(gravity).items() if all(q in ct.invariantAxesE for _, q in mqs)]


def owner(gravity=True):
    '''{(m, q): output name} and {output name: number of contributions} for the chosen decomposition.'''
    out = outputs(gravity)
    return {mq: name for name, mqs in out.items() for mq in mqs}, {name: len(mqs) for name, mqs in out.items()}
