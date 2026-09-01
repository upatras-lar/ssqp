from abc import ABC, abstractmethod


class ProblemWrapper(ABC):
    def __init__(self, horizon, dt, urdf_path):
        self.H = horizon
        self.dt = dt
        self.urdf_path = urdf_path

    @abstractmethod
    def addSquaredJointAngleCost(self, k, weight):
        pass

    @abstractmethod
    def addSquaredJointVelocityCost(self, k, weight):
        pass

    @abstractmethod
    def addSquaredControlCost(self, k, weight):
        pass

    @abstractmethod
    def addSquaredForceCost(self, k, weight):
        pass

    @abstractmethod
    def addSquaredConfigurationErrorCost(self, k, desired_config, weight):
        pass

    @abstractmethod
    def addFrameTranslationBound(self, k, frame, lb, ub):
        pass

    @abstractmethod
    def addFrameLinearVelocityBound(self, k, frame, lb, ub):
        pass

    @abstractmethod
    def addContactConstraint(self, k, frame, max_normal_force, eq_tol):
        pass

    @abstractmethod
    def finalize(self, parameter_dict):
        pass

    @abstractmethod
    def solve(self, parameter_dict):
        pass

    @abstractmethod
    def save(self, output_json):
        pass
