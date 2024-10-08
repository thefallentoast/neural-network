import math

def round_fixed(n, decimals=0): # python's round rounds down
    multiplier = 10**decimals
    return math.floor(n * multiplier + 0.5) // multiplier