import easygopigo as e3
p = e3.EasyGoPiGo3()
ds = p.init_distance_sensor()
ms = p.init_servo("SERVO2")
import time
import numpy

def Maze():
	g = numpy.empty((4,5),dtype=object)
	cpos = [0,0]
	while (s1 = False and s2 = False):
		ai = Isweep()
		s1, s2 = False, False
		if ai[1] > 22 and ai[1] < 26 and ds.read_inches() > 4:
			s1 = True
			cpos = [3,0]
			lpos = [3,1]
			g[3][2].append(("fork",False))
			g[3][2].append(("heading",180))
			g[3][1].append(("fork",False))
			g[3][1].append(("heading",180))
			p.drive_inches(24)
		elif ai[1] > 10 and ai[1] < 14 and ds.read_inches() > 4:
			s2 = True
			lpos = [2,2]
			cpos = [2,3]
			g[2][2].append(("fork",False))
			g[2][2].append(("heading",0))
			p.drive_inches(12)
		while cpos != [0,2]:
			z = ssweep(g[lpos[0],lpos[1]][-1,1])
			snextdir(z,g[lpos[0],lpos[1]][-1,1],g,cpos,s1,s2)
			smartmove(g[lpos[0],lpos[1]][-1,1],cpos,lpos,s1,s2)
		fblink()




def Isweep(): #Returns a tuple of the angle with the highest angle and also the distance which it read at that angle
	a=0
	m=numpy.zeros((45,2))
	for i in range(45):
		m[i][0] = a
		m[i][1] = ds.read_inches()
		a += 8
		p.turn_degrees(8)
		time.sleep(.2)
	p.turn_degrees(-8,)
	ac = 0
	md = 0
	for k in range(45):
		if m[k][1] > md:
			ac = m[k][0]
			md = m[k][1]
	p.turn_degrees(ac)
	return (ac,md)
def fblink():
	p.set_eye_color((0,255,0))
	p.open_eyes()
	delay(200)
	p.close_eyes()
	p.open_eyes()
	delay(200)
	p.close_eyes()
	p.open_eyes()
	delay(200)
	p.close_eyes()
def ssweep(l):
	m = numpy.zeros((4,2))
	a=0
	for p in range(4):
		m[p][0] = (a+l) % 360
		m[p][1] = ds.read_inches()
		a+=90
		p.turn_degrees(90)
	p.turn_degrees(90)
	return m 
def snextdir(m,l,g,cp,sone,stwo):
	ac = 0
	md = 0
	s=0
	if g[cp[0],cp[1]][0,1] = True:
		for p in range(4):
			if m[p][0] != (l+180) % 360 and m[p][0] != g[cp[0],cp[1]][-1,1]:
				if m[p][1] > md:
					md = m[p][1]
					ac = m[p][0]
		ac = ac - l 
		p.turn_degrees(ac)

	elif g[cp[0],cp[1]][0,1] = False:
		for p in range(4):
			if m[p][1] != (l+180) % 360:
				if m[p][1] > md:
					md = m[p][1]
					ac = m[p][0]
				if m[p][1] > 5:
					s += 1
			if s >=2 and sone = True:
				g[cp[0],cp[1]][0,1] = True
				ac = 0
			if s >=2 and stwo = True:
				g[cp[0],cp[1]][0,1] = True
				ac = 180
		ac = ac - l
		p.turn_degrees(ac)
def smartmove (h,cp,lp,sone,stwo)
	if ds.read_inches() > 12.5 and ds.read_inches() < 24:
		p.drive_inches(12)
		lp = cp 
		if h == 0:
			cp = [cp[0],cp[1]+1]
		if h == 90:
			cp = [cp[0]-1,cp[1]]
		if h == 180:
			cp = [cp[0],cp[1]-1]
		if h == 270:
			cp = [cp[0]+1,cp[1]]
	if ds.read_inches() > 26 && stwo = True:
		p.drive_inches(24)
		lp = [2,4]
		cp = [1,4]
def fblink():
	p.set_eye_color((0,255,0))
	for i in range 3:
		p.open_eyes()
		time.sleep(.2)
		p.close_eyes()