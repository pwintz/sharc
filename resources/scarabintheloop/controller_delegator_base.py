import os
from abc import ABC, abstractmethod #  AbstractBaseClass

class BaseControllerExecutableProvider(ABC):
  def __init__(self, example_dir):
    self.example_dir = os.path.abspath(example_dir)
    self.build_dir = os.path.join(self.example_dir, "build")
    os.makedirs(self.build_dir, exist_ok=True)

  @abstractmethod
  def get_controller_executable(self, build_config:dict) -> str:
    pass