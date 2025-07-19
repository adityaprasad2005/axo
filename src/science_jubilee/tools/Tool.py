from typing import Dict, Iterable, List, NamedTuple, Tuple, Union
from science_jubilee.labware.Labware import Labware, Location, Well

class ToolStateError(Exception):
    """Raise this error if the tool is in the wrong state to perform such a command."""

    pass


class ToolConfigurationError(Exception):
    """Raise this error if there is something wrong with how the tool is configured"""

    pass


class Tool:
    # TODO: Is this init supposed to take a machine?
    def __init__(self, index, name, **kwargs):
        if not isinstance(index, int) or not isinstance(name, str):
            raise ToolConfigurationError(
                "Incorrect usage: load_tool(<tool_number>, <name>, **kwargs)"
            )
        self._machine = None
        self.index = index
        self.name = name
        self.is_active_tool = False
        self.tool_offset = None

        for k, v in kwargs.items():
            setattr(self, k, v)

    def post_load(self):
        """Run any code after tool has been associated with the machine."""
        pass

    # TODO:
    # add a park tool method that every tool config can define to do things that need to be done pre or post parking
    # ex: make sure pipette has dropped tips before parking 

    def lid_on_top_error_handling(self, location: Union[Well, Tuple, Location, Labware], expected_condition: bool):
        """Raise an error if the lid is on top of the tool.""" 

        # Error handling to check if the labware at location has lid or not
        if isinstance(location, Well):
            labware = location.labware_obj
        elif isinstance(location, Location):
            well_obj = location._labware
            labware = well_obj.labware_obj 
        elif isinstance(location, Labware):
            labware = location

        if labware.has_lid_on_top == expected_condition:
            pass
        else: 
            if labware.has_lid_on_top == True:
                raise ToolStateError(f"{labware} Labware has lid on top")        # You can better this error handling with a overriding option to the user
            else:
                raise ToolStateError(f"{labware} Labware has no lid on top")      # You can better this error handling with a overriding option to the user
    


    def revert_lid_on_top(self, location: Union[Well, Tuple, Location, Labware]):
        """Revert the lid on top of the tool."""
        if isinstance(location, Well):
            labware = location.labware_obj
        elif isinstance(location, Location):
            well_obj = location._labware
            labware = well_obj.labware_obj 
        elif isinstance(location, Labware):
            labware = location
        
        labware.has_lid_on_top = not labware.has_lid_on_top


def requires_active_tool(func):
    """Decorator to ensure that a tool cannot complete an action unless it is the
    current active tool.
    """

    def wrapper(self, *args, **kwargs):
        if self.is_active_tool == False:
            raise ToolStateError(
                f"Error: Tool {self.name} is not the current `Active Tool`. Cannot perform this action"
            )
        else:
            return func(self, *args, **kwargs)

    return wrapper
