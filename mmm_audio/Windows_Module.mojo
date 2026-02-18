from mmm_audio import *
from math import exp, sin, sqrt, cos, pi

from builtin.globals import global_constant

struct Windows(Movable, Copyable):
    """Stores various window functions used in audio processing. This struct precomputes several common window types."""
    comptime size: Int64 = 2048
    comptime size_f64: Float64 = 2048.0
    comptime mask: Int = 2047 # yep, gotta make sure this is size - 1

    fn __init__(out self):
        # sel0 = MBool[1](False)
        # sel1 = MBool[1](False)
        # sinc2 = sel0.select(MFloat[2](0.0), sel1.select(
        #     MFloat[2](0.0),
        #     MFloat[2](0.0)
        # ))
        pass

    fn at_phase[window_type: Int64,interp: Int = Interp.none](self, world: World, phase: Float64, prev_phase: Float64 = 0.0) -> Float64:
        """Get window value at given phase (0.0 to 1.0) for specified window type."""

        @parameter
        if window_type == WindowType.hann:
            ref hann = global_constant[hann_window]()
            return SpanInterpolator.read[1, interp,True,self.mask](world, Span(hann), phase * self.size_f64, prev_phase * self.size_f64)
        elif window_type == WindowType.hamming:
            ref hamming = global_constant[hamming_window]()
            return SpanInterpolator.read[1, interp,True,self.mask](world, Span(hamming), phase * self.size_f64, prev_phase * self.size_f64)
        elif window_type == WindowType.blackman:
            ref blackman = global_constant[blackman_window]()
            return SpanInterpolator.read[1, interp,True,self.mask](world, Span(blackman), phase * self.size_f64, prev_phase * self.size_f64)
        elif window_type == WindowType.kaiser:
            ref kaiser = global_constant[kaiser_window]()
            return SpanInterpolator.read[1, interp,True,self.mask](world, Span(kaiser), phase * self.size_f64, prev_phase * self.size_f64)
        elif window_type == WindowType.sine:
            ref sine = global_constant[sine_window]()
            return SpanInterpolator.read[1, interp,True,self.mask](world, Span(sine), phase * self.size_f64, prev_phase * self.size_f64)
        elif window_type == WindowType.rect:
            return 1.0 
        elif window_type == WindowType.tri:
            return 1-2*abs(phase - 0.5)
        else:
            print("Windows.at_phase: Unsupported window type")
            return 0.0

