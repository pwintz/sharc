from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import solve_ivp
from scarabintheloop.utils import assert_is_column_vector

class Dynamics(ABC):
    def __init__(self, config):
        self.config = config
        self.setup_system()

    @abstractmethod
    def setup_system(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

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

    def evolveState(self, t0, x0, u, tf, w=None):
        """
        Evolve the state from t0 to tf given the initial state x0 and control input u.

        Parameters:
        - t0: Initial time
        - x0: Initial state
        - u: Control input
        - tf: Final time
        - w: Disturbance

        Returns:
        - t: Time points
        - x: State trajectory
        """
        
        assert_is_column_vector(x0, 'x0')
        n = x0.shape[0]

        # Make sure the shapes are consistent
        x0 = x0.squeeze()   

        # Make u and w column vectors.
        assert_is_column_vector(u, 'u')
        if w:
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
    
    def getDynamicsFunction(self):
        return self.evolveState