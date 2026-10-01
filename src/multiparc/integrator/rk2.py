from functools import partial

from multiparc.integrator.explicit_rk import ExplicitRK

Midpoint = partial(
    ExplicitRK,
    bt_time=[0.0, 0.5],
    bt_state=[[], [0.5]],
    bt_final=[0.0, 1.0],
)

Heun = partial(
    ExplicitRK,
    bt_time=[0.0, 1.0],
    bt_state=[[], [1.0]],
    bt_final=[0.5, 0.5],
)

Ralston = partial(
    ExplicitRK,
    bt_time=[0.0, 2.0 / 3.0],
    bt_state=[[], [2.0 / 3.0]],
    bt_final=[0.25, 0.75],
)
