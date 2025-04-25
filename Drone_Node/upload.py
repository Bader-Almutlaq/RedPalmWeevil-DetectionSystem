import json
import socket
import json_utils
import io
import traceback
from time import sleep
import pdb


SERVER_IP = "192.168.4.193"
PORT = 12345

def send_results_to_server():
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		try:
			s.connect((SERVER_IP, PORT))
			print("[upload] connected server.")
		except Exception as e:
			print(f"[upload] Connection to server refused trying again.")
			return False

		try:
			# Prepare JSON data
			data = json_utils.load_json()
			json_bytes = json.dumps(data).encode('utf-8')
			json_size = len(json_bytes)
			print("[upload] JSON loaded successfully.")
		except Exception as e:
			print(f"[upload] Error in loading JSON.")
			traceback.print_exc()
			return False

		try:
			# Send JSON data
			s.sendall(json_size.to_bytes(4, byteorder='big'))
			s.sendall(json_bytes)
			print("[upload] Results JSON sent to server successfully.")
		except Exception as e:
			print(f"[upload] Error in sending JSON.")
			traceback.print_exc()
			return False

		try:
			# send collected images
			for trap_id in data["TRAPS"]:
				if(data[trap_id]["collected"]):
					image_bytes, image_size = json_utils.load_image(trap_id)
					print(f"[upload] Sending image of size {image_size} bytes for {trap_id}")
					s.sendall(image_size.to_bytes(4, byteorder='big'))
					s.sendall(image_bytes)
			print("[upload] Images sent to server successfully.")
		except Exception as e:
			print(f"[upload] Error in sending images.")
			traceback.print_exc()
			return False
	return True



def main():
	"""
	steps:
	1. check if at least one trap got collected
	2. listen for the server ip
	3. when server is found send the collected json which has all the collected trap's results
	4. send all the images collected in the drone's trip
	5. reset the collection.json file
	"""
	if(json_utils.get_flag()):
		
		if(not send_results_to_server()): # handles all the logic
			return # sending incomplete, dont reset JSON, and flag.
			
		json_utils.reset_json() # resets the json for the next collection trip
		
		print("[upload] uploaded to server successfully")
	else:
		print("[upload] no trap is collected from yet... ")
