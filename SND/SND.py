import easygopigo as e3
p = e3.EasyGoPiGo3()
ds = p.init_distance_sensor()
ms = p.init_servo("SERVO2")
import time
import numpy

def snd():
	p.drive_inches(24)
	p.turn_degrees(90)
	p.drive_inches(12)
	p.turn_degrees(-90)
	p.drive_inches(12)
	p.turn_degrees(-90)
	blink ((0,0,255))
	p.drive_inches(24)
	p.turn_degrees(90)
	p.turn_degrees(180)
	p.drive_inches(12)
	p.turn_degrees(-90)
	p.drive_inches(24)
	p.turn_degrees(-90)
	p.drive_inches(48)
	p.turn_degrees(-90)
	p.drive_inches(12)
def blink(color):
