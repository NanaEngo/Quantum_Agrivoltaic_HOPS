import pytest


def test_mesohops_api_and_version():
    """Integration smoke test for MesoHOPS API compatibility.

    This test will be skipped if `mesohops` is not installed in the test environment.
    It verifies that the expected utility functions and trajectory noise functions
    are present in the installed mesohops package.
    """
    mesohops = pytest.importorskip("mesohops")

    # Check version attribute (optional - not all mesohops versions expose this)
    version = getattr(mesohops, "__version__", None)
    if version is None:
        import warnings

        warnings.warn("mesohops does not expose __version__ attribute", UserWarning, stacklevel=2)

    # Check expected utility function exists
    util = pytest.importorskip("mesohops.util.bath_corr_functions")
    assert hasattr(util, "bcf_convert_dl_to_exp"), "bcf_convert_dl_to_exp not found"

    # Check expected trajectory exp_noise exists
    traj = pytest.importorskip("mesohops.trajectory.exp_noise")
    assert hasattr(traj, "bcf_exp"), "bcf_exp not found in mesohops.trajectory.exp_noise"
