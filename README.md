Benchmark cases for testing the NSCBC module in IC3

Tests include:

NSCBC_test_01:
    A 1D channel flow at 1m/s with an NSCBC_LaminarInflow as the inlet and NSCBC_PressureOutlet as the outlet.

    Broadband signal is applied at the inlet from a file ./BC/signal.txt.
    Signal is recorded in probes along the x direction, readings are compared to the injected signal.
    
    The test covers:
        1) injection properties of the inlet
        2) reflection properties of the outlet


NSCBC_test_02:
    Same setup as above. Signal is applied at the outlet instead.

    The test covers:
        1) injection properties of the outlet
        2) reflection properties of the inlet

NSCBC_test_03:
    Same setup as above. A strong (u_acoustic>>u_mean) square pulse signal is applied at the outlet instead of the broadband acoustic wave.

    The test covers:
        1) robustness in handling strong injected signals
        2) handling of reverse flow at the inlet

