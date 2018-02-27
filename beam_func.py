def beam_func(L, E, q):
    """Uniformly loaded simply supportd beam
    """
    I = 0.000012 #m4
    return (5*q*L**4)/(384*E*I)
