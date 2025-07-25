import json
import os
import string 
from dataclasses import dataclass 
from typing import Dict, Iterable, List, NamedTuple, Tuple, Union 

import time  
from datetime import datetime, timedelta 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

from science_jubilee.Machine import Machine
from science_jubilee.tools.Tool import Tool
from science_jubilee.decks.Deck import Deck
from science_jubilee.labware.Labware import Labware, Location, Well
# from science_jubilee.tools.Pipette import Pipette
from science_jubilee.tools.Double_Syringe import DoubleSyringe
from science_jubilee.tools.Syringe import Syringe
from science_jubilee.tools.Vacuum_Gripper import VacuumGripper 
from science_jubilee.tools.Oceandirect_axo import Spectrometer  


# Caution:[Done] While coding, keep in mind that sample_labware_sy/ssy/spec [vial_name] refer to different well objects. Hence when you apply any ops like dispense only the currentLiquidVolume paramter for that Well object will be updated. 
# Hence use only sample_labware_sy for all the single/dual syringe operations and in the spectromter operations use sample_labware_spec but update the sample_labware_sy Well object side-by-side. 

# TODO: Add the Lid handling code wherever needed with all the conditions. Do it for the record_spectrum and make_vial functions where we consider lids over the vials also and cover them after the op if completed. 
# TODO: [Done] Repair the problem with local and global copies of the self.samples2_sy (global labware class)-> sample_labware_sy(we make changes in well objects of it) -> self.samples2_sy(but the global labware class remains the same as before)
# TODO: Make the code more general and ask the user to tell where the well_number n slot_number where the precursors and solvents are kept. 

class Experiment:
    """
    A class to represent an experiment.
    """ 

    def __init__(self, machine: Machine, deck: Deck, all_tools: Dict[str, Tool], all_labwares: Dict[str, Dict[str, Labware]]):
        """
        Initialize an Experiment object. 
        :param machine: Machine object
        :type machine: Machine
        :param deck: Deck object
        :type deck: Deck
        :param all_tools: Dictionary of all tools
        :type all
        :param all_labwares: Dictionary of all labwares
        :type all_labwares: Dict[str, Dict[str, Labware]]
        """

        self.machine = machine
        self.deck = deck
        self.all_tools = all_tools
        self.all_labwares = all_labwares 

        self.init_expand() 

    
    def init_expand(self):
        """
        Function to initialise all the tools and labwares
        """ 

        # Tools
        self.single_syringe = self.all_tools["single_syringe"]
        self.dual_syringe = self.all_tools["dual_syringe"]
        self.gripper = self.all_tools["gripper"]
        self.spectrometer = self.all_tools["spectrometer"]

        # Labwares 
        self.slot1 = self.all_labwares["slot1"]
        self.slot2 = self.all_labwares["slot2"]
        self.slot3 = self.all_labwares["slot3"]
        self.slot4 = self.all_labwares["slot4"]
        self.slot5 = self.all_labwares["slot5"]

        self.precursors = self.slot1["precursors"]
        self.samples2_sy = self.slot2["samples2_sy"]
        self.samples2_ssy = self.slot2["samples2_ssy"]
        self.samples2_spec = self.slot2["samples2_spec"]
        self.solvents = self.slot3["solvents"]
        self.vacuum_location = self.slot4["vacuum_location"]
        self.samples5_sy = self.slot5["samples5_sy"]
        self.samples5_ssy = self.slot5["samples5_ssy"]
        self.samples5_spec = self.slot5["samples5_spec"]

        # some more variables
        self.make_vial_time_mins = 6 #4.35  # Max time to physically implement the make_vial function 
        self.record_spectrum_time_mins = 2 #0.7 #Max time to physically implement the record_spectrum function
        self.need_spectrum_refs = True # Flag to check if the spectrum references are already present

    def make_batch(self, file_name: str) :
        """
        Function to create a batch of samples  
        :param file_name: Name of the Json file which contains the details of the samples to be synthesized
        :type file_name: str 

        Here the procedure of synthesis n spectrum recording is as follows:
        1. Fill dual syringe with precursors 
        2. Dispense solvents in all the vials one-by-one 
        3. Dispense metal_precursors in all the vials one-by-one  
        4. Apply make_vial on all the vials one-by-one 
            4.1 dispense Organic precursor 
            4.2 mix the soln 
            4.3 Take T0 spectrum reading 
            4.4 wash the probe 
            4.5 (Optional) if some earlier vial has a next_spectrum_reading to be recorded, it will do that first and then continue on this loop
        5. Record spectrum readings at regular intervals for all the vials

        """ 
        axo = self.machine
        precursors = self.precursors
        solvents = self.solvents


        with open(file_name, "r") as f:
            self.data = json.load(f)

        # Extract data out of the json file 
        self.spectrum_record_interval_mins = self.data["spectrum_record_interval_mins"] 
        self.max_spectrum_records = self.data["max_spectrum_records"] 


        # Add assert statement that self.spectrum_record_interval_mins is greater than the estimated self.make_vial_time_mins !!
        assert self.spectrum_record_interval_mins  > self.make_vial_time_mins, "Spectrum record interval should be greater than approx make_vial time. Otherwise, the spectrum records will be missed for the initial vials of the batch."

        # Parse all the slot names from the json 
        self.slot_names = []
        for k in self.data.keys():
            k = k.lower().strip()
            if k[:4]=="slot":
                 self.slot_names.append(k) 

        # Save the vial recipes (optional)
        self.save_vial_recipes()  

        # Record the spectrometer reference spectrum if needed 
        if self.need_spectrum_refs:
            self.record_spectrum_refs() 
            self.need_spectrum_refs = False
        else: 
            pass 

        # Step 1. Fill Dual syringe with precursors
        self.fill_dual_syringe()  

        # Step 2. Dispense solvents in all the vials one-by-one
        self.dispense_solvent_all_vials() 

        # Step 3.Dispense metal_precursors in all the vials one-by-one
        self.dispense_metal_precursor_all_vials()


        # Loop over all the slots 
        for slot_name in self.slot_names: 
            slot = int(slot_name[4:]) if slot_name[4:].isdigit() else slot_name[4:]

            all_vials = self.data[slot_name]    # a dict with "A1": {}, "A2": {} ...

            # Loop over all the vials in the slot
            for vial_name, vial_param in all_vials.items(): 
                metal_precursor_vol = vial_param["metal_precursor_vol"]
                organic_precursor_vol = vial_param["organic_precursor_vol"]
                solvent_vol = vial_param["solvent_vol"]
                
                # Step 4. Dispense the organic precursor -> mix the solution -> take T0 spectrum reading -> wash the probe
                self.make_vial(slot= slot, vial_name= vial_name, metal_precursor_vol= metal_precursor_vol, organic_precursor_vol= organic_precursor_vol, solvent_vol= solvent_vol) 

                # The below func has conditions to check whether to continue the make_vial loop or record_spectrum in-between. 
                # Mostly this function wont be repeatedly looped because of the sufficient difference between the next_spectrum_recordtime of the vials. 
                is_any_reading_taken = self.record_spectrum_if_needed()

                # We need to sleep for 2 minutes after making each vial. This is an essential step so that the synthesis of vials don't stagnate after some time. Otherwise, For example after synthesising 4 vials, the synthesis of 5th vial wont happen bcz the next_record_spectrum will be repeatedly called up for the initially synthesised vials. 
                if is_any_reading_taken == False:
                    time.sleep(self.record_spectrum_time_mins*60) 
            
        # Reset the dual syringe 
        self.reset_dual_syringe()
        
        # Put lid on top of the precursors, now that all the vials have been synthesised 
        if precursors.has_lid_on_top == False: 
            self.gripper_pick_and_place(slot_from= 4, slot_to= 1)

        # Step 5. Record spectrum till the end of the experiment 
        self.record_spectrum_till_end()

    def save_vial_recipes(self):
        """
        Function is used to save all the vial recipes in a mof_recipes.json file
        """

        # Parse the chemical names from the json  
        self.metal_precursor_name = self.data["Experiment"]["metal_precursor_name"]
        self.organic_precursor_name = self.data["Experiment"]["organic_precursor_name"]
        self.solvent_name = self.data["Experiment"]["solvent_name"]

        # Save the vial recipes in a mof_recipes.json file by looping over all the vials one by one
        # Loop over all the slots 
        for slot_name in self.slot_names: 
            slot = int(slot_name[4:]) if slot_name[4:].isdigit() else slot_name[4:]

            all_vials = self.data[slot_name]    # a dict with "A1": {}, "A2": {} ...

            # Loop over all the vials in the slot
            for vial_name, vial_param in all_vials.items(): 
                # slot_and_vial_name = slot_name + "_" + vial_name
                metal_precursor_vol = vial_param["metal_precursor_vol"]
                organic_precursor_vol = vial_param["organic_precursor_vol"]
                solvent_vol = vial_param["solvent_vol"] 

                self.spectrometer.record_mof_recipe(well_id= vial_name , metal_precursor_name= self.metal_precursor_name, metal_precursor_vol_ml = metal_precursor_vol, organic_precursor_name= self.organic_precursor_name, organic_precursor_vol_ml = organic_precursor_vol, solvent_name= self.solvent_name, solvent_vol_ml= solvent_vol, additional_notes= slot_name)

    def fill_dual_syringe(self):
        """
        Function to fill the dual syringe with dye
        """
        axo = self.machine 
        dual_syringe = self.dual_syringe 
        precursors = self.precursors  

        # Lid on top error handling 
        if precursors.has_lid_on_top == True: 
            self.gripper_pick_and_place(slot_from= 1, slot_to= 4) 
        else:
            pass
        
        # Pick up the dual syringe 
        axo.pickup_tool(dual_syringe)
        print("Picked Up Dual Syringe")

        # No need for resetting, bcz the machine is by-deault in its zero position 
        # Reset the positions of the dual syringe on top of the precursor beakers
        # drive0 = dual_syringe.e0_drive
        # current_pos0 = float(dual_syringe._machine.get_position()[drive0])
        # headroom_mm0 = current_pos0 - dual_syringe.min_range
        # headroom_ml0 = headroom_mm0 / dual_syringe.mm_to_ml
        # dual_syringe.dispense_e0(vol= headroom_ml0, sample_loc_e=precursors[1], refill_loc_e=precursors[0], s=500)
        # current_pos = float(dual_syringe._machine.get_position()[drive0])
        # print("Dual Syringe Drive 0 reset and position:", current_pos)

        # drive1 = dual_syringe.e1_drive
        # current_pos1 = float(dual_syringe._machine.get_position()[drive1])
        # headroom_mm1 = current_pos1 - dual_syringe.min_range
        # headroom_ml1 = headroom_mm1 / dual_syringe.mm_to_ml
        # dual_syringe.dispense_e1(vol= headroom_ml1, sample_loc_v=precursors[0], refill_loc_v=precursors[0], s=500)
        # current_pos = float(dual_syringe._machine.get_position()[drive1])
        # print("Dual Syringe Drive 1 reset and position:", current_pos)

        # Refill both the syringe one by one
        dual_syringe.refill(drive = dual_syringe.e0_drive, refill_loc = precursors[0].top(-54), s = 100)
        dual_syringe.refill(drive = dual_syringe.e1_drive, refill_loc = precursors[0].top(-54), s = 100)
        print("Dual Syringe Refilled") 

        # Wait a bit to let the drops fall off
        time.sleep(10)

        # Park the dual syringe
        axo.park_tool()
        print("Parked Dual Syringe") 

        # Place lid over the precursors 
        if precursors.has_lid_on_top == False: 
            self.gripper_pick_and_place(slot_from= 4, slot_to= 1)  
    
    def reset_dual_syringe(self):
        """
        Function to reset the dual syringe
        """
        axo = self.machine 
        dual_syringe = self.dual_syringe 
        precursors = self.precursors 

        # Lid on top error handling 
        if precursors.has_lid_on_top == True: 
            self.gripper_pick_and_place(slot_from= 1, slot_to= 4) 
        else:
            pass 
        
        # Pick up the dual syringe
        axo.pickup_tool(dual_syringe)
        print("Picked Up Dual Syringe")

        drive0 = dual_syringe.e0_drive
        current_pos0 = float(dual_syringe._machine.get_position()[drive0])
        headroom_mm0 = current_pos0 - dual_syringe.min_range
        headroom_ml0 = headroom_mm0 / dual_syringe.mm_to_ml
        dual_syringe.dispense_e0(vol= headroom_ml0, sample_loc_e=precursors[1], refill_loc_e=precursors[0], s=500)
        current_pos = float(dual_syringe._machine.get_position()[drive0])
        print("Dual Syringe Drive 0 reset and position:", current_pos)

        drive1 = dual_syringe.e1_drive
        current_pos1 = float(dual_syringe._machine.get_position()[drive1])
        headroom_mm1 = current_pos1 - dual_syringe.min_range
        headroom_ml1 = headroom_mm1 / dual_syringe.mm_to_ml
        dual_syringe.dispense_e1(vol= headroom_ml1, sample_loc_v=precursors[0], refill_loc_v=precursors[0], s=500)
        current_pos = float(dual_syringe._machine.get_position()[drive1])
        print("Dual Syringe Drive 1 reset and position:", current_pos) 

        # Park the dual syringe
        axo.park_tool()
        print("Parked Dual Syringe")  

        # Place lid over the precursors 
        if precursors.has_lid_on_top == False: 
            self.gripper_pick_and_place(slot_from= 4, slot_to= 1)

    def gripper_pick_and_place(self, slot_from: int, slot_to: int):
        """
        Function to pick up and place the gripper from one slot to another
        :param slot_from: Slot from which the gripper is to be picked up
        :type slot_from: int
        :param slot_to: Slot to which the gripper is to be placed
        :type slot_to: int
        """

        axo = self.machine
        gripper = self.gripper
        vacuum_location = self.vacuum_location

        axo.pickup_tool(gripper)
        print("Picked Up Vacuum Gripper")

        gripper.pick_and_place(vacuum_location[slot_from],
                            vacuum_location[slot_to], 0.7, 3)
        print(f"Picked Lid from Slot {slot_from} and Placed Lid in Slot {slot_to}")

        axo.park_tool()
        print("Parked Vacuum Gripper")  

    
    def dispense_solvent_all_vials(self):
        """
        Function to dispense solvent to all vials
        """
        axo = self.machine
        single_syringe = self.single_syringe 
        solvents = self.solvents
        precursors = self.precursors 

        # Lid on top error handling 
        if solvents.has_lid_on_top == True: 
            self.gripper_pick_and_place(slot_from= 3, slot_to= 4) 
        else:
            pass

        # Pickup the tool 
        axo.pickup_tool(single_syringe)
        print("Picked Up Single Syringe") 

        # Loop over all the slots
        for slot_name in self.slot_names: 
            slot = int(slot_name[4:]) if slot_name[4:].isdigit() else slot_name[4:]
            
            # Choose the sample labware based on the slot number
            if slot == 2: 
                sample_labware_ssy = self.samples2_ssy
                sample_labware_sy = self.samples2_sy
                sample_labware_spec = self.samples2_spec
            elif slot ==5:
                sample_labware_ssy = self.samples5_ssy 
                sample_labware_sy = self.samples5_sy
                sample_labware_spec = self.samples5_spec 

            all_vials = self.data[slot_name]    # a dict with "A1": {}, "A2": {} ...

            # Loop over all the vials in the slot
            for vial_name, vial_param in all_vials.items(): 
                metal_precursor_vol = vial_param["metal_precursor_vol"]
                organic_precursor_vol = vial_param["organic_precursor_vol"]
                solvent_vol = vial_param["solvent_vol"] 


                # Fill the vial with solvent
                single_syringe.dispense(vol= solvent_vol, sample_loc= sample_labware_ssy[vial_name].top(-10), refill_loc= solvents[1].top(-50), s= 100)
                print(f"Vial {vial_name} on Slot {slot} filled with {solvent_vol} ml solvent")
                time.sleep(10)

                # Update the currentLiquidVolume for other Well objects 
                single_syringe.update_currentLiquidVolume(volume= solvent_vol, location= sample_labware_sy[vial_name], is_dispense=True )
                single_syringe.update_currentLiquidVolume(volume= solvent_vol, location= sample_labware_spec[vial_name], is_dispense=True )

        # Reset the single syringe position before parking because we have to use it later for mix ops 
        drive = single_syringe.e_drive
        current_pos = float(single_syringe._machine.get_position()[drive])
        headroom_mm = current_pos - single_syringe.min_range
        headroom_ml = headroom_mm / single_syringe.mm_to_ml
        single_syringe.dispense(vol= headroom_ml, sample_loc=solvents[1].top(-10), refill_loc=solvents[1].top(-10), s=500)
        current_pos = float(single_syringe._machine.get_position()[drive])
        print("Single Syringe reset and position:", current_pos)
        time.sleep(5)

        # Park the single syringe
        axo.park_tool()
        print("Parked Single Syringe")   

        # Place lid over the solvents
        if solvents.has_lid_on_top == False: 
            self.gripper_pick_and_place(slot_from= 4, slot_to= 3)


    def dispense_metal_precursor_all_vials(self):
        """
        Function to dispense metal precursor to all vials
        """ 

        axo = self.machine
        dual_syringe = self.dual_syringe
        precursors = self.precursors
        solvents = self.solvents
        precursors = self.precursors  

        # Lid on top error handling 
        if precursors.has_lid_on_top == True: 
            self.gripper_pick_and_place(slot_from= 1, slot_to= 4) 
        else:
            pass 


        # Pickup the dual_syringe
        axo.pickup_tool(dual_syringe)
        print("Picked Up Dual Syringe")

        # print("Precursors[1] liquid level before dispense:", precursors[1].currentLiquidVolume)

        # Loop over all the slots
        for slot_name in self.slot_names: 
            slot = int(slot_name[4:]) if slot_name[4:].isdigit() else slot_name[4:]
            
            # Choose the sample labware based on the slot number
            if slot == 2: 
                sample_labware_ssy = self.samples2_ssy
                sample_labware_sy = self.samples2_sy
                sample_labware_spec = self.samples2_spec
            elif slot ==5:
                sample_labware_ssy = self.samples5_ssy 
                sample_labware_sy = self.samples5_sy
                sample_labware_spec = self.samples5_spec 

            all_vials = self.data[slot_name]    # a dict with "A1": {}, "A2": {} ...

            # Loop over all the vials in the slot
            for vial_name, vial_param in all_vials.items(): 
                metal_precursor_vol = vial_param["metal_precursor_vol"]
                organic_precursor_vol = vial_param["organic_precursor_vol"]
                solvent_vol = vial_param["solvent_vol"]  

                # Dispense the metal precursor into the vial
                dual_syringe.dispense_e0(vol= metal_precursor_vol, sample_loc_e=sample_labware_sy[vial_name].top(-25), refill_loc_e=precursors[0].top(-54), s=250)
                time.sleep(10)  
                print(f"Vial {vial_name} on Slot {slot} filled with {metal_precursor_vol} ml metal precursor ") 

                # print("Precursors[1] liquid level after dispense:", precursors[1].currentLiquidVolume) 

                # update the currentLiquidVolume for other Well objects
                dual_syringe.update_currentLiquidVolume(volume= metal_precursor_vol, location= sample_labware_ssy[vial_name], is_dispense=True )
                dual_syringe.update_currentLiquidVolume(volume= metal_precursor_vol, location= sample_labware_spec[vial_name], is_dispense=True ) 

        # print("Precursors[1] liquid level before reset:", precursors[1].currentLiquidVolume)

        # Reset the dual syringe:0 before parking. We wont be using it later on. 
        drive0 = dual_syringe.e0_drive
        current_pos0 = float(dual_syringe._machine.get_position()[drive0])
        headroom_mm0 = current_pos0 - dual_syringe.min_range
        headroom_ml0 = headroom_mm0 / dual_syringe.mm_to_ml
        dual_syringe.dispense_e0(vol= headroom_ml0, sample_loc_e=precursors[1], refill_loc_e=precursors[0], s=500)
        current_pos = float(dual_syringe._machine.get_position()[drive0])
        print("Dual Syringe Drive 0 reset and position:", current_pos)

        # print("Precursors[1] liquid level after reset:", precursors[1].currentLiquidVolume)

        # Park the dual syringe 
        axo.park_tool()
        print("Parked Dual Syringe")  



    def make_vial(self, slot: int, vial_name: str, metal_precursor_vol: float, organic_precursor_vol: float, solvent_vol: float):
        """
        Function to make a vial
        :param slot: Slot in which the vial is to be placed
        :type slot: int
        :param vial_name: Name of the vial
        :type vial_name: str
        :param metal_precursor_vol: Volume of metal precursor to be added to the vial
        :type metal_precursor_vol: float
        :param organic_precursor_vol: Volume of organic precursor to be added to the vial
        :type organic_precursor_vol: float
        :param solvent_vol: Volume of solvent to be added to the vial
        :type solvent_vol: float  

        This function is used to do the following routine:
        1. Dispense the organic precursor 
        2. Mix the vial 
        3. Take T0 reading 
        4. Wash the probe 
        """

        axo = self.machine
        single_syringe = self.single_syringe
        dual_syringe = self.dual_syringe
        solvents = self.solvents 
        precursors = self.precursors

        # Choose the sample labware based on the slot number
        if slot == 2: 
            sample_labware_ssy = self.samples2_ssy
            sample_labware_sy = self.samples2_sy
            sample_labware_spec = self.samples2_spec
        elif slot ==5:
            sample_labware_ssy = self.samples5_ssy 
            sample_labware_sy = self.samples5_sy
            sample_labware_spec = self.samples5_spec 


        # Pickup the dual syringe
        axo.pickup_tool(dual_syringe)
        print("Picked Up Dual Syringe")

        # Dispense organic precursor into the vial
        dual_syringe.dispense_e1(vol= organic_precursor_vol, sample_loc_v=sample_labware_sy[vial_name].top(-25), refill_loc_v=precursors[0].top(-54), s=100)
        print(f"Vial {vial_name} on Slot {slot} filled with {metal_precursor_vol} ml metal precursor and {organic_precursor_vol} ml organic precursor") 
        time.sleep(10)

        # update the currentLiquidVolume for other Well objects
        dual_syringe.update_currentLiquidVolume(volume= organic_precursor_vol, location= sample_labware_ssy[vial_name], is_dispense=True )
        dual_syringe.update_currentLiquidVolume(volume= organic_precursor_vol, location= sample_labware_spec[vial_name], is_dispense=True )

        # Park the dual syringe 
        axo.park_tool()
        print("Parked Dual Syringe")  


        # Pickup the single syringe
        axo.pickup_tool(single_syringe)
        print("Picked Up Single Syringe")

        currentLiquidVolume = sample_labware_ssy[vial_name].currentLiquidVolume 
        mix_vol = currentLiquidVolume/3

        # Mix the vial Solution 
        single_syringe.mix(vol = mix_vol, loc = sample_labware_ssy[vial_name].top(-65), num_cycles = 2, s = 300)
        print(f"Vial {vial_name} on Slot {slot} mixed") 
        time.sleep(7)

        # Park the single syringe
        axo.park_tool()
        print("Parked Single Syringe")

        # Take the T0 spectrum reading 
        self.record_spectrum(slot, vial_name)



    def nearest_spectrum_reading(self):
        """
        Returns the Slot and Vial whose next_spectrum_recordtime is nearest from the current time 
        returns:
            next_slot: int
            next_vial: str
        """ 

        """
        Strategy :  1. If the next_spectrum_recordtime for some vial is in the past, we give the highest priority to that vial. If the the next_spectrum_recordtime is too past in time from the current time, we may also consider not taking that reading
                    2. Otherwise, we give the highest priority to the vial whose next_spectrum_recordtime is in future the nearest from the current time
        """
        axo = self.machine

        # Loop through all the vials and see which one has the nearest next_spectrum_recordtime attribute 
        smallest_timediff = timedelta(minutes= 10000)
        
        # Loop through all the slots
        for slot_name in self.slot_names:
            
            # Choose the sample labware based on the slot number
            if slot_name == "slot2":
                sample_labware_ssy = self.samples2_ssy
                sample_labware_sy = self.samples2_sy 
                sample_labware_spec = self.samples2_spec
            elif slot_name == "slot5":
                sample_labware_ssy = self.samples5_ssy
                sample_labware_sy = self.samples5_sy
                sample_labware_spec = self.samples5_spec
            
            slot = int(slot_name[4:]) if slot_name[4:].isdigit() else slot_name[4:]

            all_vials = self.data[slot_name]    # a dict with "A1": {}, "A2": {} ...

            # Loop through all the vials in the slot
            for vial_name, vial_param in all_vials.items():

                vial_well_obj = sample_labware_sy[vial_name]

                if vial_well_obj.next_spectrum_recordtime is not None:
                    time_diff = vial_well_obj.next_spectrum_recordtime - datetime.now()       
                    # If somehow a vial's next_spectrum_recordtime is in the past, then its timediff will be the most negative. And accordingly it will be given the utmost priority. 

                    if time_diff < smallest_timediff:
                        smallest_timediff = time_diff
                        next_slot = slot
                        next_vial = vial_name 

            return next_slot, next_vial

    def record_spectrum_if_needed(self):
        """
        Function to record spectrum if needed 
        return bool : not flag_var 
        """ 

        flag_var = True          # This flag checks if we need to take another reading or else we are fine with only one reading
        is_any_reading_taken = False  # This flag checks if any reading has been taken till now.

        while flag_var: 
            # Break this loop if we dont need to take any more readings and now we could restart our synthesis of other vials left. 

            # Get the nearest next_spectrum_recordtime
            next_slot, next_vial = self.nearest_spectrum_reading()

            # choose the sample labware based on the slot number
            if next_slot == 2: 
                sample_labware_ssy = self.samples2_ssy
                sample_labware_sy = self.samples2_sy 
                sample_labware_spec = self.samples2_spec
            elif next_slot == 5:
                sample_labware_ssy = self.samples5_ssy
                sample_labware_sy = self.samples5_sy
                sample_labware_spec = self.samples5_spec

            # Get the Well object of the next_vial
            vial_well_obj = sample_labware_spec[next_vial]

            # Calculate the time difference between the current time and the next_spectrum_recordtime
            timediff = vial_well_obj.next_spectrum_recordtime - datetime.now()

            # Check if we can make another vial or if we urgently need to record the nearest spectrum recording 
            if timediff < timedelta(minutes = self.make_vial_time_mins): 

                while True:   # Keep running the loop until the current datetime reaches the next_spectrum_recordtime of the vial
                    time_now = datetime.now()

                    if time_now >= vial_well_obj.next_spectrum_recordtime- timedelta(seconds= 18):
                        self.record_spectrum(next_slot, next_vial) 
                        is_any_reading_taken = True
                        break
                    else:
                        time.sleep(3)  # Sleep for 3 seconds before checking again

            else:
                flag_var = False 

        return is_any_reading_taken

    def record_spectrum(self, slot, vial_name):
        """
        Function to record spectrum for a given vial
        Parameters:
            slot (int): Slot number
            vial_name (str): Vial name
        """
        axo = self.machine
        spectrometer = self.spectrometer
        solvents = self.solvents

        # Choose the sample labware based on the slot number 
        if slot == 2:
            sample_labware_sy = self.samples2_sy
            sample_labware_ssy = self.samples2_ssy
            sample_labware_spec = self.samples2_spec
        elif slot== 5:
            sample_labware_sy = self.samples5_sy
            sample_labware_ssy = self.samples5_ssy
            sample_labware_spec = self.samples5_spec


        # Lid on top error handling 
        if sample_labware_spec.has_lid_on_top == True: 
            self.gripper_pick_and_place(slot_from= slot, slot_to= 4) 
        else:
            pass 


        # Lid on top error handling 
        if solvents.has_lid_on_top == True:
            self.gripper_pick_and_place(slot_from= 3, slot_to= 4)
        else:
            pass

        # Pickup the spectrometer 
        axo.pickup_tool(spectrometer)
        print("Picked Up Spectrometer") 

        # change the num_readings_taken attribute for all the Well objects 
        sample_labware_sy[vial_name].num_readings_taken += 1
        sample_labware_ssy[vial_name].num_readings_taken += 1
        sample_labware_spec[vial_name].num_readings_taken += 1
        
        # Record the spectrum for the given vial 
        elapsed_min = (sample_labware_spec[vial_name].num_readings_taken -1)* self.spectrum_record_interval_mins

        wavelengths, vals, absorbance = spectrometer.collect_spectrum(sample_labware_spec[vial_name].top(-52), elapsed_min= elapsed_min, save= True)
        spectrometer.plot_spectrum(sample_labware_spec[vial_name].top(-52), elapsed_min= elapsed_min , show_plot=True, save_plot=True)  
        print(f"Spectrum recorded for vial {vial_name} on Slot {slot} at {elapsed_min} mins") 

        # update the next_spectrum_recordtime for all the Well objects 
        sample_labware_sy[vial_name].next_spectrum_recordtime = datetime.now() + timedelta(minutes=self.spectrum_record_interval_mins)
        sample_labware_ssy[vial_name].next_spectrum_recordtime = datetime.now() + timedelta(minutes=self.spectrum_record_interval_mins)
        sample_labware_spec[vial_name].next_spectrum_recordtime = datetime.now() + timedelta(minutes=self.spectrum_record_interval_mins)

        # Wash the probe 
        spectrometer.wash_probe(solvents[0].top(-35), n_cycles= 3)
        print("Washed Spectrometer Probe") 

        # Park the spectrometer 
        axo.park_tool() 
        print("Parked Spectrometer ") 


        # revert the lid on top of the solvents 
        if solvents.has_lid_on_top == False:
            self.gripper_pick_and_place(slot_from= 4, slot_to= 3)
        else:
            pass


    def record_spectrum_refs(self):
        """
        Function to record spectrum references
        """
        axo = self.machine
        spectrometer = self.spectrometer
        solvents = self.solvents 

        # Lid on top error handling 
        if solvents.has_lid_on_top == True: 
            self.gripper_pick_and_place(slot_from= 3, slot_to= 4) 

        axo.pickup_tool(spectrometer)
        print("Picked Up Spectrometer")

        spectrometer.position_probe(solvents[1].top(-35)) 
        print(f"Positioned Probe to collect the dark reference spectrum. \nKindly switch off the probe light")
        choice = input("Press any key to continue: ")
        print("Recording dark reference spectrum")
        spectrometer.set_dark()

        print("Kindly switch on the probe light")
        choice = input("Press any key to continue: ")
        print("Recording white reference spectrum")
        spectrometer.set_white() 

        spectrometer.configure_device()
        print("Configured Spectrometer")

        axo.park_tool() 
        print("Parked Spectrometer")

        # Reset the lid on top. Dont do this, because we run the record_spectrum after this and it requires a wash probe operation. 
        # if solvents.has_lid_on_top == False:
        #     self.gripper_pick_and_place(slot_from= 4, slot_to= 3)

    
    def record_spectrum_till_end(self):
        """
        Function to take spectrum readings for the vials till the end of the experiement
        """
        axo = self.machine
        spectrometer = self.spectrometer 

        
        while True:
            # Break the loop when all the vials have num_readings_taken >= self.max_spectrum_records 
 
            # check if all recordings have been taken or not 
            flag_break = self.all_recordings_taken()

            # If all the vials have num_readings_taken >= self.max_spectrum_records, break the loop
            if not flag_break:
                # Get the nearest next_spectrum_recordtime
                next_slot, next_vial = self.nearest_spectrum_reading() 


                # Choose the sample labware based on the slot number
                if next_slot == 2:
                    sample_labware_ssy = self.samples2_ssy
                    sample_labware_sy = self.samples2_sy 
                    sample_labware_spec = self.samples2_spec
                elif next_slot == 5:
                    sample_labware_ssy = self.samples5_ssy
                    sample_labware_sy = self.samples5_sy
                    sample_labware_spec = self.samples5_spec

                # No need for the sample_labware_spec lid on top error handling. Becuase its already there in the record_spectrum func. 

                # Get the Well object for the next_vial 
                vial_well_obj = sample_labware_spec[next_vial]

                # Calculate the time difference between the current time and the next_spectrum_recordtime
                timediff = vial_well_obj.next_spectrum_recordtime - datetime.now()

                while True:   # Keep running the loop until the current datetime reaches the next_spectrum_recordtime of the vial
                    if datetime.now() >= vial_well_obj.next_spectrum_recordtime- timedelta(seconds= 70):
                        self.record_spectrum(next_slot, next_vial)
                        break
                    else:
                        time.sleep(3)   # Wait for 3 seconds before checking again 

                # Revert the lid on top of the sample_labware. Because we need to cover the samples in between the recording of readings
                if sample_labware_spec.has_lid_on_top == False:
                    self.gripper_pick_and_place(slot_from= 4, slot_to= next_slot) 


            else:
                break 


    def all_recordings_taken(self):
        """
        Functions return True is all the recordings have been taken ; else False 
        """

        flag_break= True

        # Loop over all the slots
        for slot_name in self.slot_names: 
            slot = int(slot_name[4:]) if slot_name[4:].isdigit() else slot_name[4:]

            # Choose the sample labware based on the slot number
            if slot == 2:
                sample_labware_ssy = self.samples2_ssy
                sample_labware_sy = self.samples2_sy 
                sample_labware_spec = self.samples2_spec
            elif slot == 5:
                sample_labware_ssy = self.samples5_ssy
                sample_labware_sy = self.samples5_sy
                sample_labware_spec = self.samples5_spec 

            all_vials = self.data[slot_name]    # a dict with "A1": {}, "A2": {} ...

            # Check if all the vials have num_readings_taken >= self.max_spectrum_records
            for vial_name, vial_param in all_vials.items(): 
                if sample_labware_spec[vial_name].num_readings_taken < self.max_spectrum_records: 
                    flag_break = False  

        return flag_break 

                




# datas = make_batch(r"C:\Users\ADITI\Downloads\Aditya\Axo_Jubilee\science-jubilee\axo\axo_api_testing\draft_synthesis_plan.json")

# print(datas)
# print(type(datas))
# print(datas.keys()) 
# print(datas["slot5"])