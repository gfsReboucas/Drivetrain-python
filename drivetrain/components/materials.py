"""Material property containers."""

from dataclasses import dataclass

@dataclass
class Material:
    '''
    Simple class to store some properties of materials used to manufacture
    gears.
    '''
    E: float = 206.0e9          # [Pa],     Young's modulus
    nu: float = 0.3             # [-],      Poisson's ratio
    sigma_Hlim: float = 1500.0e6 # [Pa],     Allowable contact stress number
    rho: float = 7.83e3         # [kg/m**3], Density
    S_ut: float = 700.0e6       # [Pa],     Tensile strength
    S_y: float = 490.0e6        # [Pa],     Yield strength

    def __post_init__(self):
        # [Pa], Shear modulus.
        self.G = (self.E/2.0)/(1.0 + self.nu)
