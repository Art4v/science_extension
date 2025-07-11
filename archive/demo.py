''''
This is a test program to calculate an "A-score" for a given semiconductor wafer cleaning process.

This program is meant to test the concept of a score for a given semiconductor wafer cleaning process. It will later be expanded to include a more complex scoring system and a GUI.

The criteria for the score are:

'''

# create master dictionary for criteria
criteria_dict = {}

# define and weight criteria and append to master dictionar#
criteria_dict[0] = {'weight': 0.5, 'description': 'partical residue', 'score' : 0}
criteria_dict[1] = {'weight': 0.7, 'description': 'chemical residue', 'score' : 0}
criteria_dict[2] = {'weight': 1, 'description': 'surface damage', 'score' : 0}
criteria_dict[3] = {'weight': 1, 'description': 'oxide control', 'score' : 0}
criteria_dict[4] = {'weight': 1, 'description': 'reliability', 'score' : 0}

# ask for input and store in variable
criteria_dict[0]['score']  = float(input("Enter the score for partical residue (%): "))/100 * criteria_dict[0]['weight']
criteria_dict[1]['score']  = float(input("Enter the score for chemical residue (%): "))/100 * criteria_dict[1]['weight']
criteria_dict[2]['score']  = float(input("Enter the score for surface damage (%): "))/100 * criteria_dict[2]['weight']  
criteria_dict[3]['score']  = float(input("Enter the score for oxide control (%): "))/100 * criteria_dict[3]['weight']
criteria_dict[4]['score']  = float(input("Enter the score for reliability (%): "))/100 * criteria_dict[4]['weight']

for i in range(len(criteria_dict)):
    print(criteria_dict[i]['score'])

# calculate score

# print final score