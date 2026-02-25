from mmm_audio import *
from math import *

struct Euler[num_dims: Int](Copyable, Movable):
    """Simple Euler method ODE solver.

    Parameters:
        num_dims: Number of dimensions (state variables), e.g. 2 for position and velocity.
    """

    var state: InlineArray[Float64, Self.num_dims]
    var dt: Float64
    var world: World

    fn __init__(out self, world: World):
        """Initialize the Euler struct."""
        self.world = world
        self.dt = 1.0 / world[].sample_rate
        self.state = InlineArray[Float64, Self.num_dims](fill=Float64(0.0))

    fn step(mut self, derivatives: InlineArray[Float64, Self.num_dims]):
        """Perform a single Euler integration step.

        Args:
            derivatives: InlineArray of derivatives for each state variable.
        """
        for i in range(Self.num_dims):
            self.state[i] = self.state[i] + derivatives[i] * self.dt


struct RK2[num_dims: Int, fn_deriv: fn(InlineArray[Float64, num_dims]) capturing -> InlineArray[Float64, num_dims]](Copyable, Movable):
    """Runge-Kutta 2nd order ODE solver.

    Parameters:
        num_dims: Number of dimensions (state variables), e.g. 2 for position and velocity.
        fn_deriv: Function that computes derivatives given the current state.
    """

    var state: InlineArray[Float64, Self.num_dims]
    var world: World
    var dt: Float64
    var k1: InlineArray[Float64, Self.num_dims]
    var k2: InlineArray[Float64, Self.num_dims]
    var temp_state: InlineArray[Float64, Self.num_dims]

    fn __init__(out self, world: World):
        """Initialize the RK2 struct."""
        self.world = world
        self.dt = 1.0 / world[].sample_rate
        self.state = InlineArray[Float64, Self.num_dims](fill=0.0)
        self.k1 = InlineArray[Float64, Self.num_dims](fill=0.0)
        self.k2 = InlineArray[Float64, Self.num_dims](fill=0.0)
        self.temp_state = InlineArray[Float64, Self.num_dims](fill=0.0)

    fn step(mut self):
        """Perform a single RK2 integration step."""
        self.k1 = Self.fn_deriv(self.state)
        for i in range(Self.num_dims):
            self.temp_state[i] = self.state[i] + self.k1[i] * (self.dt / 2.0)

        self.k2 = Self.fn_deriv(self.temp_state)

        for i in range(Self.num_dims):
            self.state[i] = self.state[i] + self.k2[i] * self.dt


struct RK4[num_dims: Int, fn_deriv: fn(InlineArray[Float64, num_dims]) capturing -> InlineArray[Float64, num_dims]](Copyable, Movable):
    """Runge-Kutta 4th order ODE solver.

    Parameters:
        num_dims: Number of dimensions (state variables), e.g. 2 for position and velocity.
        fn_deriv: Function that computes derivatives given the current state.
    """

    var state: InlineArray[Float64, Self.num_dims]
    var world: World
    var dt: Float64
    var k1: InlineArray[Float64, Self.num_dims]
    var k2: InlineArray[Float64, Self.num_dims]
    var k3: InlineArray[Float64, Self.num_dims]
    var k4: InlineArray[Float64, Self.num_dims]
    var temp_state: InlineArray[Float64, Self.num_dims]

    fn __init__(out self, world: World):
        """Initialize the RK4 struct."""
        self.world = world
        self.dt = 1.0 / world[].sample_rate
        self.state = InlineArray[Float64, Self.num_dims](fill=Float64(0.0))
        self.k1 = InlineArray[Float64, Self.num_dims](fill=Float64(0.0))
        self.k2 = InlineArray[Float64, Self.num_dims](fill=Float64(0.0))
        self.k3 = InlineArray[Float64, Self.num_dims](fill=Float64(0.0))
        self.k4 = InlineArray[Float64, Self.num_dims](fill=Float64(0.0))
        self.temp_state = InlineArray[Float64, Self.num_dims](fill=Float64(0.0))

    fn step(mut self):
        """Perform a single RK4 integration step."""
        self.k1 = Self.fn_deriv(self.state)
        for i in range(Self.num_dims):
            self.temp_state[i] = self.state[i] + self.k1[i] * (self.dt / 2.0)

        self.k2 = Self.fn_deriv(self.temp_state)
        for i in range(Self.num_dims):
            self.temp_state[i] = self.state[i] + self.k2[i] * (self.dt / 2.0)

        self.k3 = Self.fn_deriv(self.temp_state)
        for i in range(Self.num_dims):
            self.temp_state[i] = self.state[i] + self.k3[i] * self.dt

        self.k4 = Self.fn_deriv(self.temp_state)
        for i in range(Self.num_dims):
            self.state[i] = self.state[i] + (self.k1[i] + 2.0 * self.k2[i] + 2.0 * self.k3[i] + self.k4[i]) * (self.dt / 6.0)