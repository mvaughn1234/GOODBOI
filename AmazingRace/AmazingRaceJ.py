import easygopigo as e3
p = e3.EasyGoPiGo3()
ds = p.init_distance_sensor()
ms = p.init_servo("SERVO2")
import time
import numpy

def Maze(): #Overall code for the Maze Challenge
	ai = Isweep()
	s1 = False
	s2 = False
	if ai(1) > 22 && ai(1) < 26: #Confirm with the DS
		s1 = True
	elif ai(1) < 14 && ai(1) > 10: #Confirm with the DS
		s2 = True
	p.turn_degrees(ai(0))
	Movenext()

def Isweep():
	a=0
	m=[][]
	for i in range(6):
		for j in range(2):
			m[i][j] = a
			m[i][j] = ds.read_inches()
		a += 60
		p.turn_degrees(60)
	p.turn_degrees(60)
	ac = 0
	md = 0
	for k in range(5):
		if m[k][1] > md:
			ac = m[k][0]
			md = m[k][1]
	return (ac,md)
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
def Movenext(x): # This function will move to the next grid spot (~1ft)
def Nextdir(x,y,z):
	g = sweep()
	for i in range(5):
		for j in range(2):

def sweep(): #Update local map
	w, h = 4,2
	m=[[0 for x in range(w)] for y in range(h)]
	for i in range(4):
		for j in range(2):
			m[i][j] = a
			m[i][j] = ds.read_inches()
		a += 90
		p.turn_degrees(90)
	p.turn_degrees(90)
	return m