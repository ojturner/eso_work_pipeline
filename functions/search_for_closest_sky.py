# function for the automated KMOS reduction pipeline
# want to search for the closest skyfile to use in 
# the reduction process - i.e. using the correct
# skyfile for subtraction

import numpy as np
from copy import copy

def search(names,
		   types,
		   num):

	"""
	Def: Take the names and types vectors from the reduction
	and search through for the closest associated sky_frame 
	"""
    
    # make copies of the num for running backwards and forwards
	b = copy(num)
	f = copy(num)
	# first run backwards until error or skyfile    
	try:
	    while not(types[b] == 'S'):
	        b -= 1
	    if b < 0:
	        b_index = 99
	    else:
	        b_index = abs(num - b)
	# may encounter the object boundary
	except IndexError:
	    b_index = 99
	# now run forwards until error or skyfile    
	try:
	    while not(types[f] == 'S'):
	        f += 1
	    f_index = abs(num - f)
	# may encounter the object boundary
	except IndexError:
	    f_index = 99
	# which is smaller
	if f_index < b_index:
	    return names[f]
	elif b_index < f_index:
	    return names[b]
	else:
	    raise IndexError('The indices should not be equal! Check observation list order')
