from torch import nn
from torch.utils.checkpoint import checkpoint


class ExplicitRK(nn.Module):
    def __init__(
        self,
        bt_time,
        bt_state,
        bt_final,
        use_checkpoint,
        corrector=None,
        correct_intermediate=False,
        **kwarg,
    ):
        """
        Explicit RK integration with optional corrector. Built from Butcher tableau.

        bt_time: list, time coeff in Butcher tableau
        bt_state: list of list, state coeff in Butcher tableau
        bt_final: list, coeff for computing final update in Butcher tableau
        use_checkpoint: bool, whether gradient checkpoint is used or not when calling the function to integrate
        corrector: callable, optional, default None. The corrector function, mostly used to enforce conservation laws through projections.
        correct_intermediate: bool, optional, default False. Whether corrector is applied to intermediate states.
        """
        super().__init__(**kwarg)
        self.use_checkpoint = use_checkpoint
        # Building the integrator from Butcher tableau
        self.n_stages = len(bt_time)
        assert len(bt_state) == self.n_stages
        assert len(bt_final) == self.n_stages
        for i, each in enumerate(bt_state):
            assert len(each) == i
        self.bt_time = bt_time
        self.bt_state = bt_state
        self.bt_final = bt_final
        # Corrector. Mostly for projections to enforce conservation laws
        self.corrector = corrector
        # Whether to correct intermediate state or not
        self.correct_intermediate = correct_intermediate

    def forward(self, f, t, current, step_size):
        """
        Explicit RK integration. Fixed step. As instructed in Butcher tableau.

        Args
        ----------
        f: callable, the function to be integrated
        current: tensor, the current state
        step_size: float, integration step size

        Returns
        -------
        final_state: tensor with the same shape of ```current```, the next state
        update: tensor with the same shape of ```current```, the update in this step divided by step_size
        """
        inter_tdot = []
        for i in range(self.n_stages):
            time_i = t + self.bt_time[i] * step_size
            state_i = current
            for j, each in enumerate(self.bt_state[i]):
                state_i = state_i + each * inter_tdot[j] * step_size
            if self.corrector and self.correct_intermediate:
                state_i = self.corrector(state_i)
            if self.use_checkpoint:
                tdot_i = checkpoint(f, time_i, state_i, use_reentrant=False)
            else:
                tdot_i = f(time_i, state_i)
            inter_tdot.append(tdot_i)
        # Compute final update
        update = inter_tdot[0] * self.bt_final[0]
        for i in range(1, self.n_stages):
            update = update + inter_tdot[i] * self.bt_final[i]
        final_state = current + step_size * update
        if self.corrector:
            final_state = self.corrector(final_state)
            update = (final_state - current) / step_size
        return final_state, update
