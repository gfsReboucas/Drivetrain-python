"""Shared helpers for drivetrain component modules."""

def check_key(key, dic):
    '''
    check if key is a part of any key from 

    Parameters
    ----------
    key : string
        DESCRIPTION.
    dic : dict
        DESCRIPTION.

    Returns
    -------
    val : float
        DESCRIPTION.

    '''
    val = 1.0
    for k, v in dic.items():
        if(key in k):
            val = v
        
    return val
