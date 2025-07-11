# import libraries

# create master dictionary for criteria
criteria_dict = {}

# define and weight criteria and append to master dictionary
criteria_dict[0] = {'weight': 1, 'description': 'partical residue', 'score' : 0}
criteria_dict[1] = {'weight': 1, 'description': 'chemical residue', 'score' : 0}
criteria_dict[2] = {'weight': 1, 'description': 'surface damage', 'score' : 0}
criteria_dict[3] = {'weight': 1, 'description': 'oxide control', 'score' : 0}
criteria_dict[4] = {'weight': 1, 'description': 'reliability', 'score' : 0}

# ask for input and store in variable
a  = input("Enter the score for partical residue (0-10): ")

# calculate score

# print final score