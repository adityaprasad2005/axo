import json
import os
import time
import requests
import paramiko
from typing import Tuple, Union
import cv2
import numpy as np
from flask import Flask, Response, render_template
from science_jubilee.labware.Labware import Labware, Location, Well
from science_jubilee.tools.Tool import Tool

class Cam(Tool):
    def __init__(
        self,
        index,
        name,
        pi_ip_address,
        image_folder,
        focus_height,
        light: bool = False,
        light_pin : int = None,
    ):
        super().__init__(
            index,
            name,
            ip_address=pi_ip_address,
            image_folder=image_folder,
            focus_height=focus_height,
            light=light,
            light_pin=light_pin
        )
        self.pi_ip = pi_ip_address
        self.pi_username = 'axo'
        self.pi_password = 'Sun92023'
        self.image_folder = image_folder
        self.focus_height = focus_height
        self.ssh = None 
        

    @classmethod
    def from_config(cls, machine, index, name, config_file: str, path: str = os.path.join(os.path.dirname(__file__), 'configs')):
        config = os.path.join(path, config_file)
        with open(config, 'rt') as f:
            kwargs = json.load(f)
        return cls(machine=machine, index=index, name=name, **kwargs)

    
    def connect_ssh(self):
        """Ensure a persistent SSH connection."""
        if self.ssh is None or not self.ssh.get_transport().is_active():
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(self.pi_ip, username=self.pi_username, password=self.pi_password)
            

    def close_ssh(self):
        """Close the SSH connection."""
        if self.ssh:
            self.ssh.close()
            self.ssh = None
        
    
    def capture_image(self, 
                      location: Union[Well, Tuple], 
                      light, 
                      light_intensity):
        """Capture an image from the video stream."""
        # Light Intensity should be in range of [0, 1] 
        assert 0 <= light_intensity <= 1 
        
        x, y, z = Labware._getxyz(location)
        
        # Offset correction
        x = x - 14.3
        y = y - 100.8
        
        self._machine.safe_z_movement()
        self._machine.move_to(x=x, y=y, wait=True)
        #self._machine.move_to(z=self.focus_height, wait=True)
        self._machine.move_to(z=161.52, wait=True)
        
        if light is True:
            self._machine.gcode(f"M42 P{self.light_pin} S{light_intensity}")
            #for _ in range(2):
            image = self.take_picture()
            self._machine.gcode(f"M42 P{self.light_pin} S0")
     
        else:
            image = self.take_picture()
        
        self._machine.safe_z_movement()
        
        return image

    def start_video_stream_server(self):
        """Check if the Flask server is running, and start it if necessary."""
        try:
            # Check if the server is already running
            test_url = f"http://{self.pi_ip}:5000"
            response = requests.get(test_url, timeout=5)
        
            if response.status_code == 200:
                print("Video stream server is already running.")
            return True  # Server is running, no need to start it again
        
        except requests.exceptions.RequestException:
            print("Server is not running. Starting it now...")
        
        try:
            #ssh = paramiko.SSHClient()
            #ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            #ssh.connect(self.pi_ip, username=self.pi_username, password=self.pi_password)
            self.connect_ssh()
            
            # Run `video_stream.py` in the background
            self.ssh.exec_command("python3 ~/video_stream.py")
            
            # Wait for Flask to start
            for i in range(5):  # Retry up to 5 times
                time.sleep(2)  # Wait before each retry
                try:
                    response = requests.get(test_url, timeout=3)
                    if response.status_code == 200:
                        print("Flask server started successfully.")
                        print("Give 12 second to make camera stabilize")
                        time.sleep(12)
                        return True

                except requests.exceptions.RequestException:
                    print(f"Waiting for Flask to start ({i+1}/5)")
            
            print("Flask server failed to start after multiple attempts.")
            return False
        
        except Exception as e:
            print(f"Failed to start video stream server: {e}")
            return False
        
    def stop_video_stream_server(self):
        """Remotely stop the Flask live stream server on the Raspberry Pi."""
        
        try:
            #ssh = paramiko.SSHClient()
            #ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            #ssh.connect(self.pi_ip, username=self.pi_username, password=self.pi_password)
            self.connect_ssh()
            # Kill all running instances of video_stream.py
            self.ssh.exec_command("pkill -f video_stream.py")
            time.sleep(2)  # Allow time for the process to stop
            self.close_ssh()
            print("Video stream server stopped successfully.")
        except Exception as e:
            print(f"Failed to stop video stream server: {e}")

        
    def take_picture(self, max_retries = 3, retry_delay = 3):
        """Remotely trigger image capture using video_stream.py and fetch it on Windows."""
        # Activate video_stream http:// when it executes and close the server after running this line.
        
        try:
            # Start video_stream.py on the Raspberry Pi
            #ssh.exec_command("python3 ~/video_stream.py")
            #time.sleep(5)  # Allow time for the server to start
            #self.start_video_stream_server()
            
            # Check if the server is already running
            #test_url = f"http://{self.pi_ip}:5000"
            #response = requests.get(test_url, timeout=5)
        
            #if response.status_code == 200:
            #    print("Video stream server is already running.")
            #return True  # Server is running, no need to start it again
        

        
            #if not self.start_video_stream_server():
            #    raise Exception("Failed to start the live stream server.")
            
            #ssh = paramiko.SSHClient()
            #ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            #ssh.connect(self.pi_ip, username=self.pi_username, password=self.pi_password)
            self.connect_ssh()
            
            
            # Send request to Raspberry Pi to capture an image
            capture_url = f"http://{self.pi_ip}:5000/capture_image"
            
            for attempt in range(max_retries):
                response = requests.get(capture_url)
            
                if response.status_code == 200 and "Image saved at" in response.text:
                    remote_path = response.text.split(" at ")[-1].strip()
                    local_path = os.path.join(self.image_folder, os.path.basename(remote_path))
                    self.fetch_image(remote_path, local_path)
                    self.delete_remote_image(remote_path) # Delete image after fetching
                
                # Stop video_stream.py after execution
                #ssh.exec_command("pkill -f video_stream.py")
                #ssh.close()
                
                # Convert the response content (JPEG) into an OpenCV image
                #image_array = np.frombuffer(response.content, dtype=np.uint8)
                    image = cv2.imread(local_path)
                #image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                    if image is None or image.size == 0:
                        print(f"Image capture attempt {attempt + 1} failed, retrying...")
                        time.sleep(retry_delay)
                        continue
                
                    return image
                
                print(f"Capture request failed (Attempt {attempt + 1}), retrying...")
                #timestamp = time.strftime("%Y%m%d_%H%M%S")
                
                #cv2.imwrite(f"{self.image_folder}/captured_image_{timestamp}.jpg", image)
                
                # Convert to HSV color space
                #hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                # Split HSV channels
                #h, s, v = cv2.split(hsv_image)

                # Adjust brightness (Value channel), saturation, hue
                #v = cv2.convertScaleAbs(v, alpha=1.2, beta = 10)  # Increase brightness
                #s = cv2.convertScaleAbs(s, alpha=1.2, beta = 10)  # Slightly boost color intensity
                #h= cv2.convertScaleAbs(h, alpha=1.0, beta = 10)


                # Merge the adjusted HSV channels
                #adjusted_hsv = cv2.merge((h, s, v))

                # Convert back to BGR format
                #adjusted_image = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)
                
                #return adjusted_image
                #return image
            #else:
            #    raise Exception("Failed to capture image remotely.")
            raise Exception("Failed to capture image after multiple attempts.")

        except requests.exceptions.RequestException as req_err:
            raise Exception(f"Network error capturing image: {req_err}")
        except paramiko.SSHException as ssh_err:
            raise Exception(f"SSH error capturing image: {ssh_err}")
        except Exception as e:
            raise Exception(f"Unexpected error capturing image: {e}")
        except Exception as e:
            raise Exception(f"Error capturing image remotely: {e}")
        
        
    def fetch_image(self, remote_path, local_path):
        """Fetch the captured image from Raspberry Pi to Windows using paramiko."""
        try:
            #ssh = paramiko.SSHClient()
            #ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            #ssh.connect(self.pi_ip, username=self.pi_username, password=self.pi_password)
            self.connect_ssh() 
            sftp = self.ssh.open_sftp()
            sftp.get(remote_path, local_path)
            sftp.close()
            #ssh.close()
        except Exception as e:
            raise Exception(f"Failed to fetch image: {e}")
        
    def delete_remote_image(self, remote_path):
        """Delete the captured image from Raspberry Pi after fetching it."""
        try:
            #ssh = paramiko.SSHClient()
            #ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            #ssh.connect(self.pi_ip, username=self.pi_username, password=self.pi_password)
            self.connect_ssh()
            self.ssh.exec_command(f"rm {remote_path}")
            #ssh.close()
        except Exception as e:
            raise Exception(f"Failed to delete remote image: {e}")
    
    def detect_circles(self, processed_image, dp=2.0, min_dist=360, param1=10, param2=80, min_radius=90, max_radius=160):
        
        """
        Hough Circle Transform :
        
        Args: 
        
        dp (float): Inverse resolution ratio for Hough Circle Transform.
        min_dist (int): Minimum distance between circle centers.
        param1 (int): Higher threshold for Canny edge detection.
        param2 (int): Accumulator threshold for circle detection.
        min_radius (int): Minimum radius of circles.
        max_radius (int): Maximum radius of circles.
        """
            
        circles = cv2.HoughCircles(processed_image, 
                                    cv2.HOUGH_GRADIENT, 
                                    dp = dp, 
                                    minDist = min_dist,
                                    param1 = param1, 
                                    param2 = param2, 
                                    minRadius = min_radius, 
                                    maxRadius = max_radius
                )
            
        return np.round(circles[0, :]).astype("int") if circles is not None else None 
        
            
    def detect_and_draw_wells(self, image_data):
        """
        Detects circular wells in an image and draws them on a copy of the original image.
        And then extract mean RGB values from detected circular regions.

        Parameters:
        - image: The original image (BGR format).
        - circles: Detected circles from Hough Circle Transform.

        Args:
            image_path (str): Path to the input image.
            output_dir (str): Directory to save the output image with detected wells.

        Returns:
            - str: Path to the output image with drawn circles.
            - dict: A list of dictionaries with circle number, RGB values, and color name.
        """
        #image = cv2.imread(image_data)
        
        try:
            
            # Convert to grayscale
            gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian Blur
            blurred = cv2.GaussianBlur(gray, (9, 9), 0.0)

            # Convert to binary using Otsu's Thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # First attempt: detect circles in blurred image
            circles = self.detect_circles(blurred)

            if circles is None or len(circles) != 1:
                raise ValueError("No valid circles detected in blurred image.")

        except Exception as e:
            print(f"Error or insufficient circles detected in blurred image: {e}. Retrying with binary image...")
            try:
                circles = self.detect_circles(binary)
                if circles is None or len(circles) != 1:
                    raise ValueError("No circles detected in binary image either.")
            except Exception as e:
                print(f"Final failure in detecting circles: {e}")
                circles = None

            # Draw circles if detected
        output = image_data.copy()
    
        if circles is not None and len(circles) == 1:
            x, y, r = circles[0]  # Unpacking the first detected circle
            cv2.circle(output, (x, y), r, (255, 0, 0), 2)  # Blue circle
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # Red dot at center

        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        timestamp = time.strftime("%Y%m%d_%H%M")
        
        output_path = os.path.join(self.image_folder, f"detected_wells_{timestamp}.jpg")
        cv2.imwrite(output_path, output)
        
        #rgb_results = []

        if circles is not None and len(circles) > 0:
                # Create a mask for the detected circle
            mask = np.zeros((image_data.shape[0], image_data.shape[1]), dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)

            # Mask the original image and compute mean RGB values
            masked_image = cv2.bitwise_and(image_data, image_data, mask=mask)
            b, g, r = cv2.split(masked_image)

            mean_rgb = {
                'R': np.mean(r[mask == 255]),
                'G': np.mean(g[mask == 255]),
                'B': np.mean(b[mask == 255])
            }
        else:
            # Default RGB values when no circle is detected
            print(f"Circles not detected.. setting RGB values to default 0")
            
            mean_rgb = {"R": 0, "G": 0, "B": 0}
            
        return output, mean_rgb