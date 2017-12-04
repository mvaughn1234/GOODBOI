import easygopigo as e3
p = e3.EasyGoPiGo3()
ds = p.init_distance_sensor()
ms = p.init_servo("SERVO2")
import time
import numpy

def Interpreter(): #Overall code for the  Interpreter challenge
	s = 3
	x = -1 #When x is -1, move left, when x is 1 move right
	p.drive_inches(20) #Move within 4 inches of 4th box
	p.turn_degrees(90) #Turn 90deg to the right
	ms.rotate_servo(180) #Face Servo to box
	p.drive_inches(-3) #Align camera with box
	if read() == 2:
		x = 1
		dwink(x)
	else:
		dwink(x)
	cread()
	p.stop()
	if read() >= 3:
		s = s + x*1
		fblink(s,read())
	elif read() <=2:
		dwink(x)
		cread()
		if read() >=3:
			s = s+x*2
			fblink(s,read())
		elif read() <=2:
			dwink(x)
			cread()
			if read() >=3:
				s = s+x*3
				fblink(s,read())
def next_box(): #Will move to next box (depends on reading)
	if read()==1:
		#Answer-die lies to the left
		p.drive_in(-14)
		s -= 1
	if read()==2:
		#Answer-die lies to the right
		p.drive_in(14)
		s += 1
def read(): #Will read the dice on box and return the value of the die-face
def gread(): #Will return a boolean. 1 = good reading, 0 = bad/no reading
def cread(): #Will continuously move in the correct direction until a "good reading"
	while (gread() == 0):
		p.drive_inches(x*14,0)
		if gread() == 1:
			p.stop()
			break
	p.stop()
def fblink(x,y): #Blink function (box index,reading)
	for i in range(x): #Index with red blinks
		p.set_eye_color(255,0,0)
		p.open_eyes()
		delay(200)
		p.close_eyes()
	for i in range(y):
		p.set_eye_color(0,0,255)
		p.open_eyes()
		delay(200)
		p.close_eyes()
def dwink(g):
	p.set_eye_color(255,255,255)
	if g == 1:
		p.open_right_eye()
		delay(200)
		p.close_right_eye()
	elif g==-1:
		p.open_left_eye()
		delay(200)
		p,close_left_eye()