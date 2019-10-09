

import itertools
import pandas as pd
import numpy as np
import sys


case_ = 2

if case_ == 1:
    #Case 1
    list_first = [
    "C(F)(F)F",
    "C(Cl)(Cl)Cl",
    "C(F)(F)Cl",
    "C(F)(Cl)Cl",
    ]

    list_left = ['C(']

    list_right = [
    ')(F)F',
    ')(Cl)Cl',
    ')(Cl)F',
    ]
elif case_ == 2:
    #Case2
    list_first = [
    "C(F)(F)F",
    "C(Cl)(Cl)Cl",
    "C(F)(F)Cl",
    "C(F)(Cl)Cl",
    'C(F)F',
    'C(Cl)Cl',
    'C(F)Cl',
    'CF',
    'CCl',
    ]

    list_left = ['C(']

    list_right = [
    ')(F)F',
    ')(Cl)Cl',
    ')(F)Cl',
    ')F',
    ')Cl',
    ]


elif case_ ==3:

    #Case 3
    list_first = [
    "C(F)(F)F",
    "C(F)(F)Cl",
    "C(F)(F)Br",
    "C(F)(Cl)Cl",
    "C(F)(Cl)Br",
    "C(F)(Br)Cl",
    "C(F)(Br)Br",
    "C(Cl)(Cl)Cl",
    "C(Cl)(Cl)Br",
    "C(Cl)(Br)Br",
    "C(Br)(Br)Br",
    'C(F)F',
    'C(F)Cl',
    'C(F)Br',
    'C(Cl)Cl',
    'C(Cl)Br',
    'C(Br)Br',
    'CF',
    'CCl',
    'CBr',
    ]

    list_left = ['C(']

    list_right = [
    ')(F)F',
    ')(F)Cl',
    ')(F)Br',
    ')(Cl)Cl',
    ')(Cl)Br',
    ')(Br)Br',
    ')F',
    ')Cl',
    ')Br']

elif case_ == 4:
    #Case 4
    list_first = [
    "C(F)(F)F",
    "C(F)(F)Cl",
    "C(F)(F)Br",
    "C(F)(Cl)Cl",
    "C(F)(Cl)Br",
    "C(F)(Br)Cl",
    "C(F)(Br)Br",
    "C(Cl)(Cl)Cl",
    "C(Cl)(Cl)Br",
    "C(Cl)(Br)Br",
    "C(Br)(Br)Br",
    'C(F)F',
    'C(F)Cl',
    'C(F)Br',
    'C(Cl)Cl',
    'C(Cl)Br',
    'C(Br)Br',
    'CF',
    'CCl',
    'C=CBr',
    ]

    list_left = ['C(']

    list_right = [
    ')(F)(F)F',
    ')(F)F',
    ')(F)Cl',
    ')(F)Br',
    ')(Cl)Cl',
    ')(Cl)Br',
    ')(Br)Br',
    ')F',
    ')Cl',
    ')Br']



'''
num_first = [[[0,0,0]], [[0,0,1]], [[0,0,2]], [[0,0,3]]]
num_first = [[[0,0,0]], [[0,0,1]]]

num_middle = [[0,0,0], [0,0,1], [0,0,2], [-1,0,0], [-1,0,1], [0,1,0]]
num_middle = [[0,0,1],[-1,0,0], [-1,0,1]]
num_middle = [[0,0,1]]

list_num = []
list_base = num_first

print('list_base')
print(list_base)

for num_chain in range(2,  3+1):
    list_next = []
    print('initiailize')

    for list_ in list_base:
        print('######################')
        print('list_')
        print(list_)

        print('list_[-1]')
        print(list_[-1])

        middle_candidate = num_middle
        print('list_[-1][1]')
        print(list_[-1][1])
        print('list_[-1][2]')
        print(list_[-1][2])

        if list_[-1][1] == 1:
            middle_candidate = [list_ for list_ in middle_candidate if middle_candidate_[1] != 0]

        if list_[-1][2] < 1:
            middle_candidate = [list_ for list_ in middle_candidate if list_[0] >= 0]
            list_tmp = list(itertools.product(list_, middle_candidate))

        elif list_[-1][2] >= 1:
            middle_candidate = [list_ for list_ in middle_candidate if list_[0] >= -1]
            list_tmp = list(itertools.product(list_, middle_candidate))

        print('list_tmp')
        print(list_tmp)

        list_next.append(list_tmp)
        print('list_next')
        print(list_next)

    list_base = list_next

print('list_base')
print(list_base)

sys.exit()
'''

list_result = []

for num_chain in range(2, 7 + 1):
    list_base = list_first
    for i in range(num_chain - 1):
        combination_ = list(itertools.product(list_left, list_base, list_right))
        list_base = [''.join(v) for v in combination_]

    #combination_ = list(itertools.product(list_totyu, list_out))
    #list_final = [''.join(v) for v in combination_]

    list_result.extend(list_base)

    print(len(list_base))

#    print(list_base)
#    print(list_result)

    array_ = np.array(list_result)
    array_ = array_.reshape(-1,1)

    #print(array_)

    df_ = pd.DataFrame(array_)
    df_.to_csv('candidate/case'  + str(case_) + '/Chain_'+ str(num_chain) + '_Num_' +str(len(list_result)) +   '.csv')
