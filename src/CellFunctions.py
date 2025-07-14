def cell_growth(t, tau, N0):
    return N0 * 2 ** (t/tau)