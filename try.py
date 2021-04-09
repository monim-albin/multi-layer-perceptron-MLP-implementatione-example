"""
by: Abdulmoinm Albinhashim
"""

import numpy as np

def diff_activation(activation_fnuction_name, previous, a_values):
    """
        return the derivation of given activation function on x
    	loss_fnuction_name -- activation function name.
    	previous -- derivation value of loss function
    	a_value -- input

    """
    # derivation of activation funcitons
    if activation_fnuction_name == "relu":
        a_values[a_values < 0 ] = 0
        a_values[a_values > 0 ] = 1
    elif activation_fnuction_name == "sigmoid":
        a_values = np.exp(a_values)
        a_values = a_values/(1 - a_values)
    # chain rule (derivation of loss function * derivation of relu)
    return previous * a_values

def new_weights(previous, h_values):
    """
        return the new weights
        previous -- the derivation of given activation function on x.
        h_values -- values of nodes after applying activation function on them

    """
    result_holder = np.array([])
    for i in previous:
        result_holder = np.append(result_holder, i * h_values, axis=0)
    return result_holder

def new_input(previous, previous_weights):
    """
        return the new inputs to new iteration in the chain rule
        previous -- the derivation of given activation function on x.
        previous_weights -- weights of current level
    """
    return previous @ previous_weights

def diff_b(previous):
    """
        return the new bias (derivation of bias is IDENTITY matrix)
        previous -- the derivation of given activation function on x.

    """
    return previous

def activation_funcion(activation_function_name, h):
    """
        return a vector after applying the given activation function
        activation_function_name -- activation function name.
        h -- linear values of all nodes the current stage before using activation function on them

    """
    if activation_function_name == "relu":
        return np.maximum(h,0)
    elif activation_function_name == "sigmoid":
         return 1/(1+np.exp(-h))
    elif activation_function_name == "new":
        return np.exp(-h**2)

""" variable initialization """

X = [1., 0.] # input
Y = [1., 1.] # output

# 1 Weight and Bias

w1 = np.array([[1.0, 2.],
               [0., 1.]])
b1 = np.array([1., 1.])

# 2 Weight and Bias

w2 = np.array([[-2., 0.],
               [1., 3.]])
b2 = np.array([-1, 1])

h=np.array(X)
a_h_list = [] #

print("\nForward Propagation\n")
for w_and_b in enumerate([(w1, b1, "sigmoid"), (w2,b2,"sigmoid")]):
    sub_h = [h, w_and_b[1][0], w_and_b[1][2]]
    # W * X + B
    h = np.matmul(w_and_b[1][0], h)
    h = h + w_and_b[1][1].T
    # save a (linear result before translated to non-linear)
    # and its h (inputs)
    sub_h.insert(0, h)
    a_h_list.insert(0, sub_h)
    print("a", w_and_b[0]+1, ": ", h)
    # relu function
    h = activation_funcion(w_and_b[1][2], h)
    print("h", w_and_b[0]+1, ": ", h)

#####################################

print("\n")
y1, y2, m= round(h[0],2), round(h[1],2), 1
# derivation of square error loss function with respect to y1
diff_y1= round(-2*Y[0] + 2*y1,2)
# derivation of square error loss function with respect to y2
diff_y2 = round(-2*Y[1] + 2*y2,2)
print("L/y1: ", diff_y1,
      "\nL/y2: ", diff_y2)

############################################

print("\nback Propagation\n")

len = len(a_h_list)
previous = np.array([diff_y1, diff_y2])
# as_item ordered as follows value before activation function, previous h, previous w
for as_items in a_h_list:
    previous = diff_activation(as_items[3], previous, as_items[0])
    print("\nderivative of ",as_items[3], " ",len,": ", previous)
    new_w = new_weights(previous, as_items[1])
    new_b = diff_b(previous)
    print("new W",len,": ", new_w)
    print("new B",len,": ", new_b)
    previous = new_input(previous, as_items[2])
    len-=1
