from abc import ABC, abstractmethod
from typing import Callable, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from sharc.utils import assert_is_column_vector

# # Function signature for evolve_state: 
# #                 evolve_state(t0,         x0,          u,         w,    tf)->(        tf,         xf)
# EveolveStateFnc = Callable[[float, np.ndarray, np.ndarray, np.ndarray, float], Tuple[float, np.ndarray]]

class Dynamics(ABC):
    def __init__(self, config):
        self.config = config
        self.n = self.config["system_parameters"]["state_dimension"]
        self.m = self.config["system_parameters"]["input_dimension"]
        self.p = self.config["system_parameters"]["output_dimension"]
        self.w_dim = self.config["system_parameters"]["exogenous_input_dimension"]
        self.setup_system()

    def setup_system(self):
      pass

    @abstractmethod
    def evolve_state(self, t0: float, x0: np.ndarray, u: np.ndarray, w: np.ndarray, tf: float):
        """
        Evolve the state from t0 to tf given the initial state x0, control input u, and exogenous input w.

        Parameters:
        - t0: Initial time
        - x0: Initial state
        - u: Control input
        - tf: Final time
        - w: Exogenous input

        Returns:
        - t: Final time
        - x: Final state
        """
        pass

    
    def get_exogenous_input(self, t):
      """
      Define a default exogenous input function.
      To implement this in subclasses, generate an exogenous input vector "w" that is a 
      numpy column vector. 
      """
      return np.zeros((self.w_dim, 1))
    
    
class OdeDynamics(Dynamics):

    def __init__(self, config):
       super().__init__(config)

    @abstractmethod
    def system_derivative(self, t, x, u, w):
        """
        Compute the derivative of the state.
        
        Parameters:
        - t: Current time
        - x: Current state
        - u: Control input
        - w: Disturbance
        
        Returns:
        - dx/dt: Derivative of the state (make sure to return the same dimension as x)
        """
        pass

    def evolve_state(self, t0: float, x0: np.ndarray, u: np.ndarray, w: np.ndarray, tf: float):
        """
        Evolve the state from t0 to tf given the initial state x0 and control input u.

        Parameters:
        - t0: Initial time
        - x0: Initial state
        - u: Control input
        - tf: Final time
        - w: Disturbance

        Returns:
        - t: Final time
        - x: Final state
        """
        
        assert_is_column_vector(x0, 'x0')
        n = x0.shape[0]

        # Make sure the shapes are consistent.
        x0 = x0.squeeze()   

        # Make u and w column vectors.
        assert_is_column_vector(u, 'u')
        if w is not None:
          assert_is_column_vector(w, 'w')
        
        def f(t, x):    
          # Make x back into a column vector.
          x = x.reshape([n, 1]) 
          dxdt = self.system_derivative(t, x, u, w)
          assert x.shape == dxdt.shape
          dxdt = dxdt.squeeze()
          assert dxdt.ndim == 1, f'dxdt={dxdt} must be a 1D array.'
          return dxdt

        assert x0.ndim == 1, f'x0 must be a 1D array for solve_ivp. x0.shape={x0.shape}.'
        sol = solve_ivp(f, [t0, tf], x0, t_eval=np.linspace(t0, tf, 100))
        assert sol.y.shape[0] == n
        (t, x) = (sol.t[-1], sol.y[:,-1])

        x = x.reshape([n, 1])
        assert x.shape == (n, 1)
        return (t, x)