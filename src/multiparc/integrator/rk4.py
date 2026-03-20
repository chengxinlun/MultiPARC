from functools import partial
from multiparc.integrator.explicit_rk import ExplicitRK


RK4 = partial(
    ExplicitRK,
    bt_time=[0.0, 0.5, 0.5, 1.0],
    bt_state=[[], [0.5], [0.0, 0.5], [0.0, 0.0, 1.0]],
    bt_final=[1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
)

RK4_38 = partial(
    ExplicitRK,
    bt_time=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
    bt_state=[[], [1.0 / 3.0], [-1.0 / 3.0, 1.0], [1.0, -1.0, 1.0]],
    bt_final=[1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0],
)
