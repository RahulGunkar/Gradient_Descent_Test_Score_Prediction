from numpy import *


#calculate mean square error
#error = (1/N)*E (y - (mx+b))^2			#E = summnation..i.e sigma
def calculate_error_for_line_given_points(b,m,points):
	totalError = 0

	for i in range(0,len(points)):
		x = points[i , 0]
		y = points[i , 1]
		totalError += ( y - (m*x + b))**2
	return totalError/float(len(points))

def step_gradient(b_current, m_current, points , a):
	#gradient descent
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	#calculating gradient at each point
	for i in range(0, len(points)):
		x = points[i,0]
		y = points[i,1]
		b_gradient += -(2/N) * ( y - ((m_current*x) + b_current))
		m_gradient += -(2/N) *x* ( y - ((m_current*x) + b_current))

	new_b = b_current - ( a * b_gradient)
	new_m = m_current - ( a * m_current)
	return [new_b , new_m]




def gradient_descent_runner(points , initial_m , initial_b, learning_rate, num_iterations):
	b = initial_b	# 0
	m = initial_m	#0
	a = learning_rate 

	for i in range(num_iterations):
		b,m = step_gradient(b,m,array(points), a)
	return [b,m]


def run():
	points = genfromtxt('data.csv' , delimiter=',')#x , y points
	learning_rate = 0.0001 #hyperparameters
	#y = mx + b
	initial_b = 0
	initial_m = 0

	num_iterations = 1000

	print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, calculate_error_for_line_given_points(initial_b, initial_m, points) ) )
	print(" Running...")
	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
	print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, calculate_error_for_line_given_points(b, m, points)))



if __name__ == '__main__':
	run()
