from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import solve_ivp

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
        def ode(t, x):                
            return self.system_derivative(t, x, u, w)
        
        # Make sure the shapes are consistent
        if len(x0.shape) > 1:
            x0 = x0.reshape(-1)
        if len(u.shape) > 1:
            u = u.reshape(-1)
            
        sol = solve_ivp(ode, [t0, tf], x0, t_eval=np.linspace(t0, tf, 100))
        (t,x) = (sol.t[-1], sol.y[:,-1])

        assert x.shape == x0.shape, (x.shape, x0.shape)
        return (t, x)
    
    def getDynamicsFunction(self):
        return self.evolveState