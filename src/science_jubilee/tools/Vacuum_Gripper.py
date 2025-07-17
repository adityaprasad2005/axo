import json
import os
from typing import Tuple, Union

from science_jubilee.labware.Labware import Labware, Location, Well
from science_jubilee.tools.Tool import (
    Tool,
    ToolConfigurationError,
    ToolStateError,
    requires_active_tool
)

class VacuumGripper(Tool):
    def __init__(
        self,
        index: int,
        name: str,
        vacuum_pin : int = None,
        limit_switch_pin : int = None
    ):
        super().__init__(index, name, vacuum_pin=vacuum_pin)
        if vacuum_pin is None:
            raise ToolConfigurationError("VacuumGripper requires a vacuum_pin to be specified")
        if limit_switch_pin is None:
            raise ToolConfigurationError("VacuumGripper requires a limit_switch_pin to be specified")
        
        self.vacuum_pin = vacuum_pin
        self.limit_switch_pin = limit_switch_pin
        
        
    @requires_active_tool
    def grip(self, 
             location : Union[Well, Tuple, Location], 
             pwm : int,
             retract_z_after_probe : float = 3.0):
        
        assert 0 <= pwm <= 1
        
        x, y, z = Labware._getxyz(location)
        
        self._machine.safe_z_movement()
        self._machine.move_to(x = x, y = y, wait = True)
        
        # # Activate vacuum before probing
        self._machine.gcode(f"M42 P{self.vacuum_pin} S{pwm}")
        
        # Trigger probing to Z-stop attached to gripper, S-1 to avoid z-offset changes 
        self._probe_limit_switch(retract_z_after_probe)
        
    
    @requires_active_tool 
    def drop(self, 
             location : Union[Well, Tuple, Location],
             retract_z_after_probe : float = 0.0):
        
        x, y, z = Labware._getxyz(location)
        
        self._machine.safe_z_movement()
        self._machine.move_to(x = x, y = y, wait = True)
        self._probe_limit_switch(retract_z_after_probe)
        
        self._machine.gcode(f"M42 P{self.vacuum_pin} S0")
    
    def _probe_limit_switch(self, retract_z_after_probe : float = 0.0):
        """Triggers limit switch probing and retracts Z afterwards."""
        self._machine.gcode(f"G30 K{self.limit_switch_pin} S-1")
        if retract_z_after_probe > 0:
            self._machine.move(dz=retract_z_after_probe, wait=True)
       
    @requires_active_tool
    def pick_and_place(self, 
                       grip : Union[Well, Tuple, Location], 
                       drop : Union[Well, Tuple, Location],
                       pwm,
                       retract_z_after_probe):
        
        self.grip(grip, pwm, retract_z_after_probe)
        self.drop(drop)
        self._machine.safe_z_movement()
        