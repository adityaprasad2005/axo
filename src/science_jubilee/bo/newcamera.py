import json
import os
import webcolors
import time
import paramiko
from typing import Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
#import requests

from science_jubilee.labware.Labware import Labware, Location, Well
from science_jubilee.tools.Tool import Tool, requires_active_tool

# Set up SSH client
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Replace with your Raspberry Pi's IP and SSH credentials
pi_ip = '192.168.1.2'
pi_username = 'axo'
pi_password = 'Sun92023'

class Cam(Tool):
    def __init__(
        self,
        index,
        name,
        pi_ip_address,
        pi_password,
        pi_username,
        image_folder,
        focus_height,
        light: bool = False,
        light_pin : int = None,
    ):
    
        super().__init__(
            index,
            name,
            ip_address=pi_ip_address,
            password = pi_password,
            pi_username = pi_username,
            image_folder=image_folder,
            light=light,
            light_pin=light_pin,
            focus_height=focus_height,
        )
        
        # self.tool_offset = self._machine.tool_z_offsets[self.index] 
        # Replace with your Raspberry Pi's IP and SSH credentials
        self.ip = "192.168.1.2"
        self.pi_username = 'axo'
        self.pi_password = 'Sun92023'
    
    
    @classmethod
    def from_config(cls, machine, index, name, config_file: str,
                    path :str = os.path.join(os.path.dirname(__file__), 'configs')):
        config = os.path.join(path,config_file)
        with open(config, 'rt') as f:
            kwargs = json.load(f)
        return cls(machine=machine, index=index, name=name,**kwargs)
        
    
    def capture_image(
        self, 
        location: Union[Well, Tuple],
        save_dir,  
        timeout = 30):
        """Capture image from Raspberry Pi UVC camera and write to the file
        
        : param timeout :
        """
        # Light Intensity should be 0 
        # assert 0 <= light_intensity <= 1 
        
        x, y, z = Labware._getxyz(location)
        
        # Offset correction
        x = x-14.3
        y = y-100.8
        
        self._machine.safe_z_movement()
        self._machine.move_to(x=x, y=y, wait = True)
            
        # Location to take the picture
        #picture_height = 40 - abs(self.tool_offset)
        self._machine.move_to(z=180, wait = True)
        
        #if light is True:
        #    self._machine.gcode(f"M42 P{self.light_pin} S{light_intensity}")
        #    image = self.take_picture(timeout=timeout)
        #    self._machine.gcode(f"M42 P{self.light_pin} S0")
        #else:
        image = self.take_picture(save_dir)
        
        return image
    
    #@requires_active_tool    
    def take_picture(self, save_dir):    
        
        if not isinstance(save_dir, str):
            raise TypeError(f"Expected save_dir to be a string, got {type(save_dir)}")
        
        try:
            # Set up SSH client 
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect('192.168.1.2', username = 'axo', password = 'Sun92023')

            # Python Script to capture an image
            capture_script = """
import cv2
import time
# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open webcam")
# Set explicit camera parameters to match video streaming quality
cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # Set exposure (if supported by the camera)
#cap.set(cv2.CAP_PROP_GAIN, 0)       # Set gain
cap.set(cv2.CAP_PROP_BRIGHTNESS, 200)  # Adjust brightness
cap.set(cv2.CAP_PROP_CONTRAST, 50)  # Adjust contrast
cap.set(cv2.CAP_PROP_SATURATION, -10)  # Adjust saturation
#cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4000)  # Set white balance (blue)
#cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 4500)   # Set white balance (red)
# Let the camera stabilize

# Capture a single frame
ret, frame = cap.read()
if not ret:
    raise Exception("Failed to capture image")
# Save the captured frame
cv2.imwrite('/tmp/image.jpg', frame)
# Release the camera
cap.release()
            """
            
            # Write the script to the Raspberry Pi
            sftp = ssh.open_sftp()
            
            with sftp.file('/tmp/capture_image.py', 'w') as f:
                f.write(capture_script)
            sftp.close()

            
            # Run the command to capture the image using fswebcam
            stdin, stdout, stderr = ssh.exec_command("python3 /tmp/capture_image.py")
            stdout.channel.recv_exit_status()  # Wait for the command to complete
            
            # Fetch the captured image from the Raspberry Pi to the local system
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
                save_path = os.path.join(save_dir, 'captured_image.jpg')
            else:
                save_path = 'captured_image.jpg'  # Default to the current directory
            
            sftp = ssh.open_sftp()
            
            #with sftp.file('/tmp/image.jpg', 'rb') as f:
            #    image_data = f.read()
                
            sftp.get('/tmp/image.jpg', save_path)  # Save to current directory on Windows
            
            # Clean up temporary files on the Raspberry Pi
            ssh.exec_command("rm /tmp/capture_image.py /tmp/image.jpg")
            ssh.close()
        
        except paramiko.SSHException as e:
            print(f"Error capturing image via SSH: {e}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
        
        # Attribute image into np.array
        image = cv2.imread(save_path)
        
        return image
    
    def detect_and_draw_wells(self, image_data, output_dir, dp=1.5, min_dist=360, param1=80, param2=100, min_radius=30, max_radius=200):
        """
        Detects circular wells in an image and draws them on a copy of the original image.
        And then extract mean RGB values from detected circular regions.

        Parameters:
        - image: The original image (BGR format).
        - circles: Detected circles from Hough Circle Transform.

        Args:
            image_path (str): Path to the input image.
            output_dir (str): Directory to save the output image with detected wells.
            dp (float): Inverse resolution ratio for Hough Circle Transform.
            min_dist (int): Minimum distance between circle centers.
            param1 (int): Higher threshold for Canny edge detection.
            param2 (int): Accumulator threshold for circle detection.
            min_radius (int): Minimum radius of circles.
            max_radius (int): Maximum radius of circles.

        Returns:
            - str: Path to the output image with drawn circles.
            - dict: A list of dictionaries with circle number, RGB values, and color name.
        """
        #image = cv2.imread(image_data)

        # Convert to grayscale for circle detection
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        output = image_data.copy()

        # If circles are detected, draw them
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)  # Green circle
                cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # Red dot at the center

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, f"detected_wells.jpg")
        cv2.imwrite(output_path, output)
        
        rgb_results = []

        if circles is not None and len(circles) > 0:
            circles = np.squeeze(circles)
            if len(circles.shape) == 1:  # Handle single circle detection
                circles = [circles]
            circles = np.round(circles).astype("int")

            for i, circle in enumerate(circles):
                if len(circle) != 3:  # Ensure circle contains x, y, r
                    continue
                x, y, r = circle

                # Create a mask for the current circle
                mask = np.zeros((image_data.shape[0], image_data.shape[1]), dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)

                # Mask the original image
                masked_image = cv2.bitwise_and(image_data, image_data, mask=mask)

                # Split channels and calculate mean RGB values
                b, g, r = cv2.split(masked_image)
                mean_r = np.mean(r[mask == 255])
                mean_g = np.mean(g[mask == 255])
                mean_b = np.mean(b[mask == 255])

                # Append the result
                rgb_results.append({
                    'Circle Number': i + 1,  # Circle number matching annotation
                    'R': mean_r,
                    'G': mean_g,
                    'B': mean_b
                })

        return output, rgb_results
    
        
    def decode_image(self, image_bin):
        """Decode a bstring image into an np.array

        :param image_bin: the image as a bstring
        :type image_bin: bytes
        :return: the image as an np.array
        :rtype: np.array
        """
        image_arr = np.frombuffer(image_bin, np.uint8)
        image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
        image_rgb = image[:, :, [2, 1, 0]]
        
        return image_rgb
    
    def process_image(self, image_bin, radius=50):
        """Externally callable function to run processing pipeline

        :param image_bin: the image as a bstring
        :type image_bin: bytes
        :param radius: the radius (in pixels) of the circular mask, defaults to 50
        :type radius: int, optional
        :return: the average rgb values of the masked image
        :rtype: list

        """
        image = self.decode_image(image_bin)
        r = radius
        masked_image = self._mask_image(image, r)
        t = time.time()
        cv2.imwrite(f"./sampleimage_full_{t}.jpg", image)
        cv2.imwrite(f"./sampleimage_masked_{t}.jpg", masked_image)
        rgb_values = self._get_rgb_avg(masked_image)
        return rgb_values
    
    def _circular_image(self, image, radius):
        """Apply a circular mask to an image"""
        
        # Get the dimensions of the image
        height, width = image.shape[:2]
        size = min(height, width)
        center = (size // 2, size // 2)
        
        # Resize image to fit the mask
        new_image = cv2.resize(image, (size, size))
        
        # Create circualr to fit the mask
        mask = np.zeros((size, size), dtype = "uint8")
        cv2.circle(mask, center, radius, (255,), thickness = -1)
        circular_image = cv2.bitwise_and(image, image, mask=mask)
        
        return circular_image
    
    def get_rgb_avg(self, image):
        """Extract the average rgb values from an image

        :param image: the image object
        :type image: np.array
        :return: the average rgb values in a list [R,G,B]
        :rtype: list
        """
        # Ensure the image is in the expected format (3 color channels)
        if len(image.shape) < 3 or image.shape[2] != 3:
            raise ValueError("Input image must be in RGB formate with 3 color channels.")
        
        avg_rgb = np.mean(image, axis=(0, 1))
        
        return avg_rgb.tolist()
    
    def _get_rgb_avg(self, image):
        """Extract the average rgb values from an image

        :param image: the image object
        :type image: np.array
        :return: the average rgb values in a list [R,G,B]
        :rtype: list
        """
        bgr = []
        for dim in [0, 1, 2]:
            flatdim = image[:, :, dim].flatten()
            indices = flatdim.nonzero()[0]
            value = flatdim.flatten()[indices].mean()
            bgr.append(value)

        # opencv uses bgr so convert to rgb for loss
        print("swapping")
        rgb = [bgr[i] for i in [2, 1, 0]]
        return rgb
    
    def view_image(self, image_bin, masked=False, radius=50):
        """Show the image in a matplotlib window

        :param image_bin: the image as a bstring
        :type image_bin: bytes
        :param masked: Wether to mask the image or not, defaults to False
        :type masked: bool, optional
        :param radius: the size (in pixel) of the circular mask toapply to the image , defaults to 50
        :type radius: int, optional
        """

        image = self.decode_image(image_bin)
        if masked is True:
            image = self._mask_image(image, radius)
        else:
            pass

        fig, ax = plt.subplots(figsize=(3, 4))
        plt.setp(plt.gca(), autoscale_on=True)
        ax.imshow(image)
        