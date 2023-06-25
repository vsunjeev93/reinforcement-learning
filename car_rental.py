import pandas as pd
import numpy as np
import scipy.stats as sp
from itertools import product
import random
import time
import matplotlib.pyplot as plt
''' 

s=(n1,n2) a tuple of cars in each location n1,n2<=20
a=[-5,5] +ve means 1 to 2; -ve means movement from location 2 to 1
r(reward)= no. rental requests at location 1(rr1)+ no. rental requests at location 2(rr2)
transition probability p(s',r|s,a)
where: s'=(n1',n2'), n1'=n1-a-rr1+ret1,n2'=n2+a-rr2+ret2 where ret1=return in location 1

'''
t0 = time.time()
p={}
N_cars=20
a_max=5
mu_ret1=3
mu_ret2=2
mu_req1=3
mu_req2=4
move_cost=2


N_list=range(0,N_cars+1)
# N_list1=range(0,7)
# N_list2=range(0,8)
# N_list3=range(0,7)
# N_list4=range(0,5)

N_list1=range(0,11)
N_list2=range(0,12)
N_list3=range(0,11)
N_list4=range(0,10)
V={}
pi={}
gamma=0.9
park_fee=4
for i in [1,2,3,4]:
	p_req1=sp.poisson.pmf(5,3)
	print(p_req1)
for (i,j) in product(N_list,N_list):
	V[(i,j)]=random.uniform(0,1)
	if i>j:
		pi[i,j]=0
	else:
		pi[i,j]=0
	for a in np.arange(max(-j,-a_max),min(a_max,i)+1):
		
		# print(i,j,a)
		p[(i,j,a)]={}
		# for (n_req1,n_req2,n_ret1,n_ret2) in product(N_list1,N_list2,N_list3,N_list4):
		for n_req1 in N_list1:
			p_req1=sp.poisson.pmf(n_req1,mu_req1)
			# print(p_req1,mu_req1,n_req1)
			for n_ret1 in N_list3:
				p_ret1=sp.poisson.pmf(n_ret1,mu_ret1)
				n1=i-a+n_ret1-min(i-a,n_req1)
				for n_req2 in N_list2:
					p_req2=sp.poisson.pmf(n_req2,mu_req2)
					for n_ret2 in N_list4:
						p_ret2=sp.poisson.pmf(n_ret2,mu_ret2)
						n2=j+a+n_ret2-min(j+a,n_req2)
						if a>0:
							r=-move_cost*(a-1)
						else:
							r=-abs(move_cost*a)
						r+=min(i-a,n_req1,N_cars)*10+min(j+a,n_req2,N_cars)*10
						if n2>10:
							r-=park_fee
						if n1>10:
							r-=park_fee
						# print('action',a,r)
						if n1<0:
							n1=0
						elif n1>N_cars:
							n1=N_cars
						if n2<0:
							n2=0
						elif n2>N_cars:
							n2=N_cars
						# print(i,j,a,n1,n2,r)
						p[(i,j,a)][(n1,n2,r)]=p[(i,j,a)].get((n1,n2,r),0)+p_ret1*p_ret2*p_req1*p_req2
sum_p={}
for (i,j) in product(N_list,N_list):
	for a in np.arange(max(-j,-a_max),min(a_max,i)+1):
		sum_p[(i,j,a)]=0
		for entry in p[(i,j,a)].keys():
			sum_p[(i,j,a)]+=p[(i,j,a)][tuple(entry)]
		print('sum_p',sum_p[(i,j,a)])


		# if a==0 and j!=0:
		# 	print(p[(i,j,a)])
		# 	sys.exit(0)

#policy evaluation
evaluate=True
improve=True
iterate=True
while iterate:
	while evaluate:
		delta=0
		for (i,j) in product(N_list,N_list):
			v=V[(i,j)]
			a=pi[(i,j)]
			s=0
			for (i_,j_,r) in p[(i,j,a)].keys():
				s+=p[i,j,a][i_,j_,r]*(r+gamma*V[(i_,j_)])
			V[(i,j)]=s
			delta=max(delta,abs(V[(i,j)]-v))
			# print(delta,V[(i,j)])
		if delta<1/100:
			evaluate=False
			break
	improve=True
	stable=True
	while improve:
		for (i,j) in product(N_list,N_list):
			max_v=-float('inf')
			a_m=pi[(i,j)]
			for a in np.arange(max(-j,-a_max),min(a_max,i)+1):
				v=0
				for (i_,j_,r) in p[(i,j,a)].keys():
					v+=p[i,j,a][i_,j_,r]*(r+gamma*V[(i_,j_)])
				# print('action',a,v)
				if v>max_v:
					max_v=v
					a_m=a
			print(i,j,a_m,'new policy',stable)
			if a_m!=pi[(i,j)]:
				stable=False
			pi[(i,j)]=a_m
		if stable:
			iterate=False
			improve=False
		else:
			evaluate=True
			improve=False




from scipy.interpolate import interp2d

# f will be a function with two arguments (x and y coordinates),
# but those can be array_like structures too, in which case the
# result will be a matrix representing the values in the grid 
# specified by those arguments
x_list=[]
y_list=[]
z_list=[]
for ((x,y),a) in pi.items():
	x_list.append(x)
	y_list.append(y)
	z_list.append(a)

f = interp2d(x_list,y_list,z_list,kind="linear")

x_coords = np.arange(min(x_list),max(x_list)+1)
y_coords = np.arange(min(y_list),max(y_list)+1)
t1 = time.time()
print('total time',t1-t0)
Z = f(x_coords,y_coords)

fig = plt.imshow(Z,
           extent=[min(x_list),max(x_list),min(y_list),max(y_list)],
           origin="lower")

# Show the positions of the sample points, just to have some reference
fig.axes.set_autoscale_on(False)
plt.scatter(y_list,x_list,400,facecolors='none')

plt.show()

# for (i,j) in product(N_list,N_list):











