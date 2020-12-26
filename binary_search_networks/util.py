# Author: Isamu Isozaki, Shesh Dave
# Date: 2020/11/10
# Purpose: Utility function folders
from binary_search_networks.binary_search_parser import binary_search_parser
import numpy as np
import matplotlib.pyplot as plt
import random
# Input: list of arguments
# Output: dictionary of arguments
# Parses the arguments
# def parse_arguments(args):
#     arg_parser = binary_search_parser()
#     args, _ = arg_parser.parse_known_args(args)
#     args = vars(args)
#     return args


"""
@param arr1 : Input array
@param n    : Target element
@param None
linear_search in an array
"""


def linear_search(arr1, n):
    arr2 = []
    duplicate = False
    for i in range(len(arr1)):
        if n == arr1[i]:
            duplicate = True
            arr2.append(i)
    if duplicate == True:
        print("n value found at index: ")
        for i in arr2:
            print(i)
    else:
        print("n value not found")


def generate_random_noise(amplitude):
    return (2*random.random()-1)*amplitude

# Input: Integer, n, number of points. Float, p, the proportional std of the point
# Output: List of length n which is a distribution which is a cusp
# Parses the arguments


def get_cusp(n, a=1, x=0.5, gamma=0.0, p=2):
    """
    n: number of points
    a: amplitude
    x: ratio of the peak in the range
    gamma: randomness
    p: power
    y = a(i-nx)^p + noise(gamma) where i from 0 to n
    """
    if not (0 < x < 1):
        return
    # For the cusp function, this function is y = ax^2  for the first half of n and a(x-n)^2 for the next half
    output = []
    # first half
    for i in range(int(n*x)):
        output.append(a*i**p + generate_random_noise(gamma))

    # first half
    for i in range(int(n*x), n):
        output.append(a*((n-i)*x/(1-x))**p + generate_random_noise(gamma))

    return output


if __name__ == "__main__":
    # arr1 = [34,23,5,6,7,11,2,23,8,94,40,61,23,5]
    # print(arr1)
    # n = int(input("enter n value: "))
    # search(arr1, n)
    y1 = get_cusp(n=10, a=0.3, gamma=20, x=0.7, p=2)
    plt.plot(y1)
    # y2 = get_cusp(n=100000, a=0.7, gamma=0.01, x=0.7, p=.7)
    # plt.plot(y1)

    plt.show()
