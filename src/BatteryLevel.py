from djitellopy import Tello
import os

tello = Tello()
tello.connect()
os.system('cls')
print("-------------------------")
print(tello.get_battery())
print("-------------------------")