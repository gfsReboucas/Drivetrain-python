"""Configuration dataclasses for drivetrain model inputs."""

from dataclasses import dataclass

from drivetrain.dynamics import torsional_2DOF


@dataclass
class DrivetrainConfig:
    """Passive container for optional drivetrain constructor inputs."""

    N_st: int = 3
    stage: list | None = None
    P_rated: float = 5.0e3
    n_rated: float = 12.1
    main_shaft: object | None = None
    m_Rotor: float = 110.0e3
    J_Rotor: float = 57231535.0
    m_Gen: float = 1900.0
    J_Gen: float = 534.116
    dynamic_model: object = torsional_2DOF
