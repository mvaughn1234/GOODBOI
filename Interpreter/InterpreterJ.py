import easygopigo as e3
p = e3.EasyGoPiGo3()
ds = p.init_distance_sensor()
ms = p.init_servo("SERVO2")
import time
import numpy
import DieReader
from picamera import PiCamera
import cv2

def Interpreter(): #Overall code for the  Interpreter challenge
	die = DieReader.DieReader()
	cam = PiCamera()
	cam.resolution = (480,480)
	cam.start_preview()
	sleep(2)

	s = 3
	x = -1 #When x is -1, move left, when x is 1 move right
	p.drive_inches(20) #Move within 4 inches of 4th box
	p.turn_degrees(90) #Turn 90deg to the right
	ms.rotate_servo(180) #Face Servo to box
	p.drive_inches(-3) #Align camera with box
	if read(die,cam) == 2:
		x = 1
		dwink(x)
	else:
		dwink(x)
	cread(die,cam)
	p.stop()
	if read(die,cam) >= 3:
		s = s + x*1
		fblink(s,read(die,cam))
	elif read(die,cam) <=2:
		dwink(x)
		cread(die,cam)
		if read(die,cam) >=3:
			s = s+x*2
			fblink(s,read(die,cam))
		elif read(die,cam) <=2:
			dwink(x)
			cread(die,cam)
			if read(die,cam) >=3:
				s = s+x*3
				fblink(s,read(die,cam))
def next_box(die,cam): #Will move to next box (depends on reading)
	if read(die,cam)==1:
		#Answer-die lies to the left
		p.drive_in(-14)
		s -= 1
	if read(die,cam)==2:
		#Answer-die lies to the right
		p.drive_in(14)
		s += 1
def read(die,cam): #Will read the dice on box and return the value of the die-face
	cam.capture('pic.png')
	img = cv2.imread('pic.png')
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	num = die.get_num(img)
	return num


def cread(die,cam): #Will continuously move in the correct direction until a "good reading"
	while (read(die,cam) == -1):
		p.drive_inches(x*14,0)
		if read(die,cam) >= 0:
			p.stop()
			break
	p.stop()
def fblink(x,y): #Blink function (box index,reading)
	for i in range(x): #Index with red blinks
		p.set_eye_color((255,0,0))
		p.open_eyes()
		time.sleep(.2)
		p.close_eyes()
	for i in range(y):
		p.set_eye_color((0,0,255))
		p.open_eyes()
		time.sleep(.2)
		p.close_eyes()
def dwink(g):
	p.set_eye_color((255,255,255))
	if g == 1:
		p.open_right_eye()
		time.sleep(.2)
		p.close_right_eye()
	elif g==-1:
		p.open_left_eye()
		time.sleep(.2)
		p,close_left_eye()

Interpreter()