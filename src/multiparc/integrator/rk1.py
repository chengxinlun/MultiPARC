from functools import partial
from multiparc.integrator.explicit_rk import ExplicitRK


Euler = partial(
    ExplicitRK,
    bt_time=[0.0],
    bt_state=[[]],
    bt_final=[1.0],
)
