import builtins

import pytest


def test_gearset_has_no_kisssoft_method():
    from drivetrain.gears import GearSet

    assert not hasattr(GearSet, "KISSsoft")


def test_kisssoft_adapter_imports_without_comtypes():
    from drivetrain.adapters import kisssoft

    assert kisssoft.create_kisssoft_gearset is not None


def test_kisssoft_adapter_reports_missing_comtypes(monkeypatch):
    from drivetrain.adapters import kisssoft

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "comtypes.client":
            raise ImportError("No module named 'comtypes'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="KISSsoft integration requires"):
        kisssoft._load_create_object()
