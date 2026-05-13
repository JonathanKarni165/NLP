import numpy as np
'''
softmax in a numerically stable way. 
softmax(x_i) = exp(x_i) / sum(exp(x_j)) 

Args: vec -> Real vector to perform softmax on
Returns: softmax vector from vec (softmax(vec))
'''
def softmax(vec):
    c = -np.max(vec)
    stab_vec = vec + c
    exp_vec = np.exp(stab_vec)
    return exp_vec / np.sum(exp_vec)

if __name__ == "__main__":
    vec = np.array([1, 200, 3])
    i = 1
    print(softmax(vec, i), ' hi')