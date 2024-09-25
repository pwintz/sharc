import numpy as np
from dynamics_base import Dynamics

class LTIDynamics(Dynamics):
    def setup_system(self):
        self.A = self.config["system_parameters"]["A"]
        self.B = self.config["system_parameters"]["B"]
        self.B_dist = self.config["system_parameters"]["B_dist"]
        
        # Do we need this in dynamics?
        self.C = self.config["system_parameters"]["C"]
        self.D = self.config["system_parameters"]["D"]

    def system_derivative(self, t, x, u, w):
        if w is None:
            w = np.array([0])
        dxdt = self.A @ x + self.B @ u + self.B_dist @ w
        assert dxdt.shape == x.shape, (dxdt.shape, x.shape, u.shape, w.shape, (self.A @ x).shape, (self.B_dist @ w).shape)
        return dxdt
    
class ACCDynamics(LTIDynamics):
    # Similar to LTI system but matrices are dependent on tau
    def setup_system(self):
        tau = self.config["system_parameters"]["tau"]
        self.A = np.array([
            [0, 1, 0, 0, 0],
            [0, -1/tau, 0, 0, 0],
            [1, 0, 0, -1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, -1/tau]
        ])
        self.B = np.array([
            [0],
            [0],
            [0],
            [0],
            [-1/tau]
        ])
        self.B_dist = np.array([
            [0],
            [1/tau],
            [0],
            [0],
            [0]
        ])
        self.C = np.array([
            [0, 0, 1, 0, 0],
            [1, 0, 0, -1, 0]
        ])
        self.D = np.zeros([5, 1])

class CartPoleDynamics(Dynamics):
    def setup_system(self):
        # System parameters
        self.M = self.config["system_parameters"]["M"]
        self.m = self.config["system_parameters"]["m"]
        self.J = self.config["system_parameters"]["J"]
        self.l = self.config["system_parameters"]["l"]
        self.c = self.config["system_parameters"]["c"]
        self.gamma = self.config["system_parameters"]["gamma"]
        self.g = self.config["system_parameters"]["g"]

    def system_derivative(self, t, x, u, w):
        # Constants
        M_t = self.M + self.m
        J_t = self.J + self.m * self.l**2
        
        # Recover state parameters
        x_pos = x[0]     # position of the base
        theta = x[1]     # angle of the pendulum
        vx = x[2]        # velocity of the base
        omega = x[3]     # angular rate of the pendulum
        
        # Compute common terms
        s_t = np.sin(theta)
        c_t = np.cos(theta)
        o_2 = omega**2
        l_2 = self.l**2
        
        # Calculate derivatives
        dxdt = np.zeros(4)
        dxdt[0] = vx
        dxdt[1] = omega
        dxdt[2] = (-self.m * self.l * s_t * o_2 + self.m * self.g * (self.m * l_2 / J_t) * s_t * c_t -
                   self.c * vx - (self.gamma / J_t) * self.m * self.l * c_t * omega + u[0]) / (M_t - self.m * (self.m * l_2 / J_t) * c_t * c_t)
        dxdt[3] = (-self.m * l_2 * s_t * c_t * o_2 + M_t * self.g * self.l * s_t - self.c * self.l * c_t * vx -
                   self.gamma * (M_t / self.m) * omega + self.l * c_t * u[0]) / (J_t * (M_t / self.m) - self.m * (self.l * c_t)**2)
        
        assert dxdt.shape == x.shape, (dxdt.shape, x.shape)
        return dxdt
    
class CustomDynamics(Dynamics):
    def setup_system(self):
        # System parameters, feel free to use self.config to get the parameters you need
        # from the config files as below
        # self.M = self.config["system_parameters"]["M"]
        return
    
    def system_derivative(self, t, x, u, w):
        # Calculate and return the derivative of the state as a function of x,t,u,w
        # make sure it has the same shape as x before returning
        
        # assert dxdt.shape == x.shape, (dxdt.shape, x.shape)
        # return dxdt
        return