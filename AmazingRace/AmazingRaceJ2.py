import easygopigo as e3
p = e3.EasyGoPiGo3()
ds = p.init_distance_sensor()
ms = p.init_servo("SERVO2")
import time
import numpy

def Maze(): #Overall code for the Maze Challenge
	ai = isweep()
	finish = False
	while finish = False:
		sweep()
		finish = finish()
		fork()
		nextdir()
		movenext()
	flbink()
def isweep(): #put the vehicle in the correct direction, returns distance to wall in front
	a=0
	m=numpy.zeros((6,2))
	for i in range(6):
		for j in range(2):
			m[i][j] = a
			m[i][j] = ds.read_inches()
		a += 60
		p.turn_degrees(60)
	p.turn_degrees(60)
	ac = 0
	md = 0
	for k in range(6):
		if m[k][1] > md:
			ac = m[k][0]
			md = m[k][1]
	p.turn_degrees(ac)
def sweep(): #returns array of angle and distance measured
	m=numpy.zeros((4,2))
	for i in range(4):
		for j in range(2):
			m[i][j] = a
			m[i][j] = ds.read_inches()
		a += 90
		p.turn_degrees(90)
	p.turn_degrees(90)
	return m


def finish():


	g = sweep
	for i in range


def fork():


def nextdir():


def movenext():


def fblink():
	p.set_eye_color(0,255,0)
	p.open_eyes()
	delay(200)
	p.close_eyes()
	p.open_eyes()
	delay(200)
	p.close_eyes()
	p.open_eyes()
	delay(200)
	p.close_eyes()