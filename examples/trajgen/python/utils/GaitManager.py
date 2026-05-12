import numpy as np
import ssqpy.model.costs as costs
import ssqpy.model.constraints as constraints


class GaitManager:
    """
    GaitManager class
    Accepts a model for a quadruped and generates gaits of different types
    """

    def __init__(self, model, nq, nv, foot_frames, swing_cost_height=-1.0):
        """
        Initialize GaitManager

        :param model: The SSQP model (must not be finalized)
        :param nq: The configuration size
        :param nv: The velocity size
        :param foot_frames: Ordered foot frames: [front_left, front_right,
        rear_left, rear_right]
        """
        self.model = model
        self.fl_foot = foot_frames[0]
        self.fr_foot = foot_frames[1]
        self.rl_foot = foot_frames[2]
        self.rr_foot = foot_frames[3]
        self.nq = nq
        self.nv = nv
        self.swing_cost_height = swing_cost_height

        self.fl_foot_vel_bound = constraints.FrameLinearVelocityBound(
            model, self.fl_foot, np.zeros(3), np.zeros(3)
        )

        self.fr_foot_vel_bound = constraints.FrameLinearVelocityBound(
            model, self.fr_foot, np.zeros(3), np.zeros(3)
        )

        self.rl_foot_vel_bound = constraints.FrameLinearVelocityBound(
            model, self.rl_foot, np.zeros(3), np.zeros(3)
        )

        self.rr_foot_vel_bound = constraints.FrameLinearVelocityBound(
            model, self.rr_foot, np.zeros(3), np.zeros(3)
        )

        self.fl_foot_ag_bound = constraints.FramePositionBound(
            model,
            self.fl_foot,
            np.array((-np.inf, -np.inf, 0.0)),
            np.array((np.inf, np.inf, np.inf)),
        )

        self.fl_foot_ag_cost = costs.FrameSquaredTranslationErrorCost(
            model,
            self.fl_foot,
            np.array((0.0, 0.0, swing_cost_height)),
            np.array((0.0, 0.0, 1.0)),
        )

        self.fr_foot_ag_bound = constraints.FramePositionBound(
            model,
            self.fr_foot,
            np.array((-np.inf, -np.inf, 0.0)),
            np.array((np.inf, np.inf, np.inf)),
        )

        self.fr_foot_ag_cost = costs.FrameSquaredTranslationErrorCost(
            model,
            self.fr_foot,
            np.array((0.0, 0.0, swing_cost_height)),
            np.array((0.0, 0.0, 1.0)),
        )

        self.rl_foot_ag_bound = constraints.FramePositionBound(
            model,
            self.rl_foot,
            np.array((-np.inf, -np.inf, 0.0)),
            np.array((np.inf, np.inf, np.inf)),
        )

        self.rl_foot_ag_cost = costs.FrameSquaredTranslationErrorCost(
            model,
            self.rl_foot,
            np.array((0.0, 0.0, swing_cost_height)),
            np.array((0.0, 0.0, 1.0)),
        )

        self.rr_foot_ag_bound = constraints.FramePositionBound(
            model,
            self.rr_foot,
            np.array((-np.inf, -np.inf, 0.0)),
            np.array((np.inf, np.inf, np.inf)),
        )

        self.rr_foot_ag_cost = costs.FrameSquaredTranslationErrorCost(
            model,
            self.rr_foot,
            np.array((0.0, 0.0, swing_cost_height)),
            np.array((0.0, 0.0, 1.0)),
        )

    def addStandingStage(self, k):
        self.model.addContactConstraint(k, self.fl_foot)
        self.model.addConstraints(k, self.fl_foot_vel_bound)
        self.model.addContactConstraint(k, self.fr_foot)
        self.model.addConstraints(k, self.fr_foot_vel_bound)
        self.model.addContactConstraint(k, self.rl_foot)
        self.model.addConstraints(k, self.rl_foot_vel_bound)
        self.model.addContactConstraint(k, self.rr_foot)
        self.model.addConstraints(k, self.rr_foot_vel_bound)

        return [self._getInitControl(4)]

    def setWalk(self, start_stage, end_stage, cycle_steps=40):
        self._checkInput(start_stage, end_stage, cycle_steps)

        init_controls = []
        for k in range(start_stage, end_stage + 1):
            gait_stage = (4 * k // cycle_steps) % 4

            if gait_stage == 0:
                self.model.addContactConstraint(k, self.fr_foot)
                self.model.addConstraints(k, self.fr_foot_vel_bound)
                self.model.addContactConstraint(k, self.rl_foot)
                self.model.addConstraints(k, self.rl_foot_vel_bound)
                self.model.addContactConstraint(k, self.rr_foot)
                self.model.addConstraints(k, self.rr_foot_vel_bound)

                self.model.addConstraints(k, self.fl_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fl_foot_ag_cost)

            elif gait_stage == 1:
                self.model.addContactConstraint(k, self.fl_foot)
                self.model.addConstraints(k, self.fl_foot_vel_bound)
                self.model.addContactConstraint(k, self.fr_foot)
                self.model.addConstraints(k, self.fr_foot_vel_bound)
                self.model.addContactConstraint(k, self.rl_foot)
                self.model.addConstraints(k, self.rl_foot_vel_bound)

                self.model.addConstraints(k, self.rr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.rr_foot_ag_cost)

            elif gait_stage == 2:
                self.model.addContactConstraint(k, self.fl_foot)
                self.model.addConstraints(k, self.fl_foot_vel_bound)
                self.model.addContactConstraint(k, self.rl_foot)
                self.model.addConstraints(k, self.rl_foot_vel_bound)
                self.model.addContactConstraint(k, self.rr_foot)
                self.model.addConstraints(k, self.rr_foot_vel_bound)

                self.model.addConstraints(k, self.fr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fr_foot_ag_cost)

            elif gait_stage == 3:
                self.model.addContactConstraint(k, self.fl_foot)
                self.model.addConstraints(k, self.fl_foot_vel_bound)
                self.model.addContactConstraint(k, self.fr_foot)
                self.model.addConstraints(k, self.fr_foot_vel_bound)
                self.model.addContactConstraint(k, self.rr_foot)
                self.model.addConstraints(k, self.rr_foot_vel_bound)

                self.model.addConstraints(k, self.rl_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.rl_foot_ag_cost)

            init_controls.append(self._getInitControl(3))

        return init_controls

    def setAmble(self, start_stage, end_stage, cycle_steps=40):
        self._checkInput(start_stage, end_stage, cycle_steps)

        init_controls = []
        for k in range(start_stage, end_stage + 1):
            gait_stage = (4 * k // cycle_steps) % 4

            if gait_stage == 0:
                self.model.addContactConstraint(k, self.fr_foot)
                self.model.addConstraints(k, self.fr_foot_vel_bound)
                self.model.addContactConstraint(k, self.rl_foot)
                self.model.addConstraints(k, self.rl_foot_vel_bound)

                self.model.addConstraints(k, self.fl_foot_ag_bound)
                self.model.addConstraints(k, self.rr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fl_foot_ag_cost)
                    self.model.addCost(k, self.rr_foot_ag_cost)

            elif gait_stage == 1:
                self.model.addContactConstraint(k, self.fl_foot)
                self.model.addConstraints(k, self.fl_foot_vel_bound)
                self.model.addContactConstraint(k, self.rl_foot)
                self.model.addConstraints(k, self.rl_foot_vel_bound)

                self.model.addConstraints(k, self.fr_foot_ag_bound)
                self.model.addConstraints(k, self.rr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fr_foot_ag_cost)
                    self.model.addCost(k, self.rr_foot_ag_cost)

            elif gait_stage == 2:
                self.model.addContactConstraint(k, self.fl_foot)
                self.model.addConstraints(k, self.fl_foot_vel_bound)
                self.model.addContactConstraint(k, self.rr_foot)
                self.model.addConstraints(k, self.rr_foot_vel_bound)

                self.model.addConstraints(k, self.fr_foot_ag_bound)
                self.model.addConstraints(k, self.rl_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fr_foot_ag_cost)
                    self.model.addCost(k, self.rl_foot_ag_cost)

            elif gait_stage == 3:
                self.model.addContactConstraint(k, self.fr_foot)
                self.model.addConstraints(k, self.fr_foot_vel_bound)
                self.model.addContactConstraint(k, self.rr_foot)
                self.model.addConstraints(k, self.rr_foot_vel_bound)

                self.model.addConstraints(k, self.fl_foot_ag_bound)
                self.model.addConstraints(k, self.rl_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fl_foot_ag_cost)
                    self.model.addCost(k, self.rl_foot_ag_cost)

            init_controls.append(self._getInitControl(2))

        return init_controls

    def setTrot(self, start_stage, end_stage, cycle_steps=40):
        self._checkInput(start_stage, end_stage, cycle_steps)

        init_controls = []
        for k in range(start_stage, end_stage + 1):
            gait_stage = k % cycle_steps

            if gait_stage < 0.3 * cycle_steps:
                self.model.addContactConstraint(k, self.fl_foot)
                self.model.addConstraints(k, self.fl_foot_vel_bound)
                self.model.addContactConstraint(k, self.rr_foot)
                self.model.addConstraints(k, self.rr_foot_vel_bound)

                self.model.addConstraints(k, self.fr_foot_ag_bound)
                self.model.addConstraints(k, self.rl_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fr_foot_ag_cost)
                    self.model.addCost(k, self.rl_foot_ag_cost)

                init_controls.append(self._getInitControl(2))
            elif gait_stage > 0.5 * cycle_steps and gait_stage < 0.8 * cycle_steps:
                self.model.addContactConstraint(k, self.fr_foot)
                self.model.addConstraints(k, self.fr_foot_vel_bound)
                self.model.addContactConstraint(k, self.rl_foot)
                self.model.addConstraints(k, self.rl_foot_vel_bound)

                self.model.addConstraints(k, self.fl_foot_ag_bound)
                self.model.addConstraints(k, self.rr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fl_foot_ag_cost)
                    self.model.addCost(k, self.rr_foot_ag_cost)

                init_controls.append(self._getInitControl(2))
            else:
                self.model.addConstraints(k, self.fl_foot_ag_bound)
                self.model.addConstraints(k, self.fr_foot_ag_bound)
                self.model.addConstraints(k, self.rl_foot_ag_bound)
                self.model.addConstraints(k, self.rr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fl_foot_ag_cost)
                    self.model.addCost(k, self.fr_foot_ag_cost)
                    self.model.addCost(k, self.rl_foot_ag_cost)
                    self.model.addCost(k, self.rr_foot_ag_cost)

                init_controls.append(self._getInitControl(0))

        return init_controls

    def setPace(self, start_stage, end_stage, cycle_steps=60):
        self._checkInput(start_stage, end_stage, cycle_steps)

        init_controls = []
        for k in range(start_stage, end_stage + 1):
            gait_stage = k % cycle_steps

            if gait_stage < 0.3 * cycle_steps:
                self.model.addContactConstraint(k, self.fl_foot)
                self.model.addConstraints(k, self.fl_foot_vel_bound)
                self.model.addContactConstraint(k, self.rl_foot)
                self.model.addConstraints(k, self.rl_foot_vel_bound)

                self.model.addConstraints(k, self.fr_foot_ag_bound)
                self.model.addConstraints(k, self.rr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fr_foot_ag_cost)
                    self.model.addCost(k, self.rr_foot_ag_cost)

                init_controls.append(self._getInitControl(2))
            elif gait_stage > 0.5 * cycle_steps and gait_stage < 0.8 * cycle_steps:
                self.model.addContactConstraint(k, self.fr_foot)
                self.model.addConstraints(k, self.fr_foot_vel_bound)
                self.model.addContactConstraint(k, self.rr_foot)
                self.model.addConstraints(k, self.rr_foot_vel_bound)

                self.model.addConstraints(k, self.fl_foot_ag_bound)
                self.model.addConstraints(k, self.rl_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fl_foot_ag_cost)
                    self.model.addCost(k, self.rl_foot_ag_cost)

                init_controls.append(self._getInitControl(2))
            else:
                self.model.addConstraints(k, self.fl_foot_ag_bound)
                self.model.addConstraints(k, self.fr_foot_ag_bound)
                self.model.addConstraints(k, self.rl_foot_ag_bound)
                self.model.addConstraints(k, self.rr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fl_foot_ag_cost)
                    self.model.addCost(k, self.fr_foot_ag_cost)
                    self.model.addCost(k, self.rl_foot_ag_cost)
                    self.model.addCost(k, self.rr_foot_ag_cost)

                init_controls.append(self._getInitControl(0))

        return init_controls

    def setCanter(self, start_stage, end_stage, cycle_steps=40):
        self._checkInput(start_stage, end_stage, cycle_steps)

        init_controls = []
        for k in range(start_stage, end_stage + 1):
            gait_stage = (4 * k // cycle_steps) % 4

            if gait_stage == 0:
                self.model.addContactConstraint(k, self.rr_foot)
                self.model.addConstraints(k, self.rr_foot_vel_bound)

                self.model.addConstraints(k, self.fl_foot_ag_bound)
                self.model.addConstraints(k, self.fr_foot_ag_bound)
                self.model.addConstraints(k, self.rl_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fl_foot_ag_cost)
                    self.model.addCost(k, self.fr_foot_ag_cost)
                    self.model.addCost(k, self.rl_foot_ag_cost)

                init_controls.append(self._getInitControl(1))
            elif gait_stage == 1:
                self.model.addContactConstraint(k, self.fr_foot)
                self.model.addConstraints(k, self.fr_foot_vel_bound)
                self.model.addContactConstraint(k, self.rl_foot)
                self.model.addConstraints(k, self.rl_foot_vel_bound)

                self.model.addConstraints(k, self.fl_foot_ag_bound)
                self.model.addConstraints(k, self.rr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fl_foot_ag_cost)
                    self.model.addCost(k, self.rr_foot_ag_cost)

                init_controls.append(self._getInitControl(2))
            elif gait_stage == 2:
                self.model.addContactConstraint(k, self.fl_foot)
                self.model.addConstraints(k, self.fl_foot_vel_bound)

                self.model.addConstraints(k, self.fr_foot_ag_bound)
                self.model.addConstraints(k, self.rl_foot_ag_bound)
                self.model.addConstraints(k, self.rr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fr_foot_ag_cost)
                    self.model.addCost(k, self.rl_foot_ag_cost)
                    self.model.addCost(k, self.rr_foot_ag_cost)

                init_controls.append(self._getInitControl(1))
            elif gait_stage == 3:
                self.model.addConstraints(k, self.fl_foot_ag_bound)
                self.model.addConstraints(k, self.fr_foot_ag_bound)
                self.model.addConstraints(k, self.rl_foot_ag_bound)
                self.model.addConstraints(k, self.rr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fl_foot_ag_cost)
                    self.model.addCost(k, self.fr_foot_ag_cost)
                    self.model.addCost(k, self.rl_foot_ag_cost)
                    self.model.addCost(k, self.rr_foot_ag_cost)

                init_controls.append(self._getInitControl(0))

        return init_controls

    def setGallop(self, start_stage, end_stage, cycle_steps=40):
        self._checkInput(start_stage, end_stage, cycle_steps)

        init_controls = []
        for k in range(start_stage, end_stage + 1):
            gait_stage = (5 * k // cycle_steps) % 5

            if gait_stage == 0:
                self.model.addContactConstraint(k, self.rl_foot)
                self.model.addConstraints(k, self.rl_foot_vel_bound)

                self.model.addConstraints(k, self.fl_foot_ag_bound)
                self.model.addConstraints(k, self.fr_foot_ag_bound)
                self.model.addConstraints(k, self.rr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fl_foot_ag_cost)
                    self.model.addCost(k, self.fr_foot_ag_cost)
                    self.model.addCost(k, self.rr_foot_ag_cost)

                init_controls.append(self._getInitControl(1))
            elif gait_stage == 1:
                self.model.addContactConstraint(k, self.rl_foot)
                self.model.addConstraints(k, self.rl_foot_vel_bound)
                self.model.addContactConstraint(k, self.rr_foot)
                self.model.addConstraints(k, self.rr_foot_vel_bound)

                self.model.addConstraints(k, self.fl_foot_ag_bound)
                self.model.addConstraints(k, self.fr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fl_foot_ag_cost)
                    self.model.addCost(k, self.fr_foot_ag_cost)

                init_controls.append(self._getInitControl(2))
            elif gait_stage == 2:
                self.model.addContactConstraint(k, self.fl_foot)
                self.model.addConstraints(k, self.fl_foot_vel_bound)
                self.model.addContactConstraint(k, self.rl_foot)
                self.model.addConstraints(k, self.rl_foot_vel_bound)
                self.model.addContactConstraint(k, self.rr_foot)
                self.model.addConstraints(k, self.rr_foot_vel_bound)

                self.model.addConstraints(k, self.fr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fr_foot_ag_cost)

                init_controls.append(self._getInitControl(3))
            elif gait_stage == 3:
                self.model.addContactConstraint(k, self.fl_foot)
                self.model.addConstraints(k, self.fl_foot_vel_bound)
                self.model.addContactConstraint(k, self.fr_foot)
                self.model.addConstraints(k, self.fr_foot_vel_bound)

                self.model.addConstraints(k, self.rl_foot_ag_bound)
                self.model.addConstraints(k, self.rr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.rl_foot_ag_cost)
                    self.model.addCost(k, self.rr_foot_ag_cost)

                init_controls.append(self._getInitControl(2))
            elif gait_stage == 4:
                self.model.addConstraints(k, self.fl_foot_ag_bound)
                self.model.addConstraints(k, self.fr_foot_ag_bound)
                self.model.addConstraints(k, self.rl_foot_ag_bound)
                self.model.addConstraints(k, self.rr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fl_foot_ag_cost)
                    self.model.addCost(k, self.fr_foot_ag_cost)
                    self.model.addCost(k, self.rl_foot_ag_cost)
                    self.model.addCost(k, self.rr_foot_ag_cost)

                init_controls.append(self._getInitControl(0))

        return init_controls

    def setRotatoryGallop(self, start_stage, end_stage, cycle_steps=40):
        self._checkInput(start_stage, end_stage, cycle_steps)

        init_controls = []
        for k in range(start_stage, end_stage + 1):
            gait_stage = (8 * k // cycle_steps) % 8

            if gait_stage == 0:
                self.model.addContactConstraint(k, self.rr_foot)
                self.model.addConstraints(k, self.rr_foot_vel_bound)

                self.model.addConstraints(k, self.fl_foot_ag_bound)
                self.model.addConstraints(k, self.fr_foot_ag_bound)
                self.model.addConstraints(k, self.rl_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fl_foot_ag_cost)
                    self.model.addCost(k, self.fr_foot_ag_cost)
                    self.model.addCost(k, self.rl_foot_ag_cost)

                init_controls.append(self._getInitControl(1))
            elif gait_stage == 1:
                self.model.addContactConstraint(k, self.rl_foot)
                self.model.addConstraints(k, self.rl_foot_vel_bound)
                self.model.addContactConstraint(k, self.rr_foot)
                self.model.addConstraints(k, self.rr_foot_vel_bound)

                self.model.addConstraints(k, self.fl_foot_ag_bound)
                self.model.addConstraints(k, self.fr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fl_foot_ag_cost)
                    self.model.addCost(k, self.fr_foot_ag_cost)

                init_controls.append(self._getInitControl(2))
            elif gait_stage == 2:
                self.model.addContactConstraint(k, self.rl_foot)
                self.model.addConstraints(k, self.rl_foot_vel_bound)

                self.model.addConstraints(k, self.fl_foot_ag_bound)
                self.model.addConstraints(k, self.fr_foot_ag_bound)
                self.model.addConstraints(k, self.rr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fl_foot_ag_cost)
                    self.model.addCost(k, self.fr_foot_ag_cost)
                    self.model.addCost(k, self.rr_foot_ag_cost)

                init_controls.append(self._getInitControl(1))
            elif gait_stage == 3:
                self.model.addConstraints(k, self.fl_foot_ag_bound)
                self.model.addConstraints(k, self.fr_foot_ag_bound)
                self.model.addConstraints(k, self.rl_foot_ag_bound)
                self.model.addConstraints(k, self.rr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fl_foot_ag_cost)
                    self.model.addCost(k, self.fr_foot_ag_cost)
                    self.model.addCost(k, self.rl_foot_ag_cost)
                    self.model.addCost(k, self.rr_foot_ag_cost)

                init_controls.append(self._getInitControl(0))
            elif gait_stage == 4:
                self.model.addContactConstraint(k, self.fl_foot)
                self.model.addConstraints(k, self.fl_foot_vel_bound)

                self.model.addConstraints(k, self.fr_foot_ag_bound)
                self.model.addConstraints(k, self.rl_foot_ag_bound)
                self.model.addConstraints(k, self.rr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fr_foot_ag_cost)
                    self.model.addCost(k, self.rl_foot_ag_cost)
                    self.model.addCost(k, self.rr_foot_ag_cost)

                init_controls.append(self._getInitControl(1))
            elif gait_stage == 5:
                self.model.addContactConstraint(k, self.fl_foot)
                self.model.addConstraints(k, self.fl_foot_vel_bound)
                self.model.addContactConstraint(k, self.fr_foot)
                self.model.addConstraints(k, self.fr_foot_vel_bound)

                self.model.addConstraints(k, self.rl_foot_ag_bound)
                self.model.addConstraints(k, self.rr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.rl_foot_ag_cost)
                    self.model.addCost(k, self.rr_foot_ag_cost)

                init_controls.append(self._getInitControl(2))
            elif gait_stage == 6:
                self.model.addContactConstraint(k, self.fr_foot)
                self.model.addConstraints(k, self.fr_foot_vel_bound)

                self.model.addConstraints(k, self.fl_foot_ag_bound)
                self.model.addConstraints(k, self.rl_foot_ag_bound)
                self.model.addConstraints(k, self.rr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fl_foot_ag_cost)
                    self.model.addCost(k, self.rl_foot_ag_cost)
                    self.model.addCost(k, self.rr_foot_ag_cost)

                init_controls.append(self._getInitControl(1))
            elif gait_stage == 7:
                self.model.addConstraints(k, self.fl_foot_ag_bound)
                self.model.addConstraints(k, self.fr_foot_ag_bound)
                self.model.addConstraints(k, self.rl_foot_ag_bound)
                self.model.addConstraints(k, self.rr_foot_ag_bound)
                if self.swing_cost_height > 0:
                    self.model.addCost(k, self.fl_foot_ag_cost)
                    self.model.addCost(k, self.fr_foot_ag_cost)
                    self.model.addCost(k, self.rl_foot_ag_cost)
                    self.model.addCost(k, self.rr_foot_ag_cost)

                init_controls.append(self._getInitControl(0))

        return init_controls

    def _checkInput(self, start_stage, end_stage, cycle_steps):
        if (
            start_stage < 0
            or end_stage >= self.model.horizon
            or start_stage >= end_stage
        ):
            raise ValueError(f"Invalid stage range {start_stage} - {end_stage}")

        if cycle_steps < 0 or cycle_steps > end_stage - start_stage:
            raise ValueError(f"Invalid cycle steps value {cycle_steps}")

    def _getInitControl(self, nforces):
        init_control = np.zeros(self.nv + 3 * nforces)

        for i in range(nforces):
            init_control[-1 - 3 * i] = 15 / nforces

        return init_control
