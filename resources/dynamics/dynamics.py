import math
import numpy as np
from scarabintheloop.dynamics_base import OdeDynamics

class LTIDynamics(OdeDynamics):
    
    def __init__(self, config):
       super().__init__(config)

    def setup_system(self):
        self.A      = self.config["system_parameters"]["A"]
        self.B      = self.config["system_parameters"]["B"]
        self.B_dist = self.config["system_parameters"]["B_dist"]
        
        # Do we need this in dynamics?
        self.C = self.config["system_parameters"]["C"]
        self.D = self.config["system_parameters"]["D"]
        
        self.n = self.config["system_parameters"]["state_dimension"]
        self.m = self.config["system_parameters"]["input_dimension"]
        self.p = self.config["system_parameters"]["output_dimension"]


    def system_derivative(self, t, x, u, w):
        """
        Implement the abstract system_derivative method from OdeDynamics.
        """
        try:
          dxdt = self.A @ x + self.B @ u
          if w is not None: # If a disturbance is given, then add its effect.
            dxdt += self.B_dist @ w
        except Exception as err:
          raise ValueError(f'Failed to calculate dx/dt with the following data: \nA: {self.A}\nx: {x}\nB: {self.B}\nu: {u}\nB_dist: {self.B_dist}\nw: {w}') from err
        assert dxdt is not None
        assert dxdt.shape == (self.n, 1), f'Expected dxdt.shape = {(self.n, 1)}, but instead dxdt.shape={dxdt.shape}.\nx.shape={x.shap}, u.shape={u.shape}, w.shape={"N/A" if w is None else w.shape}, (self.A @ x).shape={(self.A @ x).shape}, (self.B_dist @ w).shape={"N/A" if w is None else (self.B_dist @ w).shape}.' 
        return dxdt
    
class ACCDynamics(LTIDynamics):
    
    def __init__(self, config):
       super().__init__(config)

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
        
        self.n = self.config["system_parameters"]["state_dimension"]
        self.m = self.config["system_parameters"]["input_dimension"]
        self.p = self.config["system_parameters"]["output_dimension"]

class ACC2Dynamics(OdeDynamics):
    def __init__(self, config):
       super().__init__(config)

    def setup_system(self):
      self.n = self.config["system_parameters"]["state_dimension"]
      self.m = self.config["system_parameters"]["input_dimension"]
      self.p = self.config["system_parameters"]["output_dimension"]

      # System parameters
      self.beta  = self.config["system_parameters"]["beta"]              
      self.gamma = self.config["system_parameters"]["gamma"]             
      self.M     = self.config["system_parameters"]["mass"]

    def system_derivative(self, t, x, u, w):
        """
        Implement the abstract system_derivative method from OdeDynamics.
        """
        assert isinstance(x, np.ndarray), f'x must be an Numpy array. Instead, it was {type(x)}.'
        assert isinstance(u, np.ndarray), f'u must be an Numpy array. Instead, it was {type(u)}.'
        assert isinstance(w, np.ndarray), f'w must be an Numpy array. Instead, it was {type(w)}.'

        # Separate state components
        p       = x[0] # Position
        h       = x[1] # Headway
        v       = x[2] # Velocity

        # Separate input components
        F_accel = u[0]
        F_brake = u[1]

        # Exogenous Inputs
        v_front = w[0]

        # Compute friction force.
        F_friction = self.beta + self.gamma * v**2
        assert F_friction >= 0

        # Compute derivative terms
        pdot        = v
        hdot        = v_front - v
        vdot        = (1/self.M) * (F_accel - F_brake - F_friction)
        # vdot        = F_accel - F_brake - F_friction
        
        # Calculate derivatives
        dxdt = np.zeros((self.n, 1), float)
        dxdt[0] = pdot        # Position
        dxdt[1] = hdot        # Headway
        dxdt[2] = vdot        # Velocity
        
        assert dxdt.shape == x.shape, (dxdt.shape, x.shape)
        return dxdt
    
    def get_exogenous_input(self, t):
        front_velocity = 12 + 2*math.sin(t) + 1*math.sin(3.23*t) + 0.4*math.sin(12.1*t)
        return np.array([[front_velocity], [1.0]])

class CartPoleDynamics(OdeDynamics):
    
    def __init__(self, config):
       super().__init__(config)

    def setup_system(self):
        # System parameters
        self.M = self.config["system_parameters"]["M"]
        self.m = self.config["system_parameters"]["m"]
        self.J = self.config["system_parameters"]["J"]
        self.l = self.config["system_parameters"]["l"]
        self.c = self.config["system_parameters"]["c"]
        self.gamma = self.config["system_parameters"]["gamma"]
        self.g = self.config["system_parameters"]["g"]
        
        self.n = self.config["system_parameters"]["state_dimension"]
        self.m = self.config["system_parameters"]["input_dimension"]
        self.p = self.config["system_parameters"]["output_dimension"]

    def system_derivative(self, t, x, u, w):
        """
        Implement the abstract system_derivative method from OdeDynamics.
        """
        # Constants
        M_t = self.M + self.m
        J_t = self.J + self.m * self.l**2
        
        # Recover state parameters
        x_pos = x[0]     # position of the base
        theta = x[1]     # angle of the pendulum
        vx    = x[2]     # velocity of the base
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
        dxdt = dxdt.reshape(4,1)
        assert dxdt.shape == x.shape, (dxdt.shape, x.shape)
        return dxdt
    
# class CustomDynamics(OdeDynamics):
#     def setup_system(self):
#         # System parameters, feel free to use self.config to get the parameters you need
#         # from the config files as below
#         # self.M = self.config["system_parameters"]["M"]
#         return
#     
#     def system_derivative(self, t, x, u, w):
#         """
#         Implement the abstract system_derivative method from OdeDynamics.
#         """
#         # Calculate and return the derivative of the state as a function of x,t,u,w
#         # make sure it has the same shape as x before returning
#         
#         # assert dxdt.shape == x.shape, (dxdt.shape, x.shape)
#         # return dxdt
#         return