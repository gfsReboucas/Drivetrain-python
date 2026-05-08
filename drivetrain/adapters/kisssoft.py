"""Optional KISSsoft COM adapter.

This module intentionally keeps the commercial COM dependency outside the core
gear geometry classes. Importing the core package does not require KISSsoft or
comtypes; those are only needed when this adapter is called.
"""

import os

import numpy as np


def _load_create_object():
    try:
        from comtypes.client import CreateObject
    except ImportError as exc:
        raise ImportError(
            "KISSsoft integration requires the optional 'comtypes' package "
            "and a local KISSsoft COM installation. Core drivetrain models "
            "can be used without this dependency."
        ) from exc

    return CreateObject


def create_kisssoft_gearset(gear_set):
    """Create and configure a KISSsoft COM module from a GearSet object."""
    create_object = _load_create_object()
    ks = create_object("KISSsoftCOM.KISSsoft")
    ks.SetSilentMode(True)

    if gear_set.configuration == "parallel":
        ks.GetModule("Z012", False)
        std_file = "CylGearPair 1 (spur gear).Z12"
        geo_meth = False
    elif gear_set.configuration == "planetary":
        ks.GetModule("Z014", False)
        std_file = "PlanetarySet 1 (ISO6336).Z14"
        geo_meth = True
    else:
        raise ValueError("Unsupported KISSsoft gear-set configuration: {}".format(gear_set.configuration))

    file_name = os.path.join("C:\\Program Files (x86)\\KISSsoft 03-2017\\example", std_file)

    try:
        ks.LoadFile(file_name)
    except Exception as exc:
        ks.ReleaseModule()
        raise RuntimeError("Error while loading file {}.".format(file_name)) from exc

    ks.SetVar("ZS.AnzahlZwi", "{}".format(gear_set.N_p))
    ks.SetVar("ZS.Geo.mn", "{:.6f}".format(gear_set.m_n))
    ks.SetVar("ZP[0].a", "{:.6f}".format(gear_set.a_w))
    ks.SetVar("ZS.Geo.alfn", "{:.6f}".format(np.radians(gear_set.alpha_n)))
    ks.SetVar("ZS.Geo.beta", "{:.6f}".format(np.radians(gear_set.beta)))
    ks.SetVar("RechSt.GeometrieMeth", "{}".format(geo_meth))

    # Maximum arithmetic mean roughness for external gears according to IEC 61400-4.
    R_a = 0.8

    for idx, zz in enumerate(gear_set.z):
        ks.SetVar("ZR[{}].z".format(idx), "{}".format(np.abs(zz)))
        ks.SetVar("ZR[{}].x.nul".format(idx), "{:.6f}".format(gear_set.x[idx]))
        ks.SetVar("ZR[{}].b".format(idx), "{:.6f}".format(gear_set.b))
        ks.SetVar("ZR[{}].Tool.type".format(idx), "2")
        ks.SetVar("ZR[{}].Vqual".format(idx), "{}".format(gear_set.Q))
        ks.SetVar("ZR[{}].RAH".format(idx), "{}".format(R_a))
        ks.SetVar("ZR[{}].RAF".format(idx), "{}".format(6.0 * R_a))

        if not ks.CalculateRetVal():
            ks.ReleaseModule()
            raise RuntimeError("Error in KISSsoft calculation.")

    return ks
