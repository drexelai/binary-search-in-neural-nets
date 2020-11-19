# Author: Isamu Isozaki, Shesh, Dave 
# Date: 2020/11/10
# Purpose: Utility function folders
from binary_search_networks.binary_search_parser import binary_search_parser 

# Input: list of arguments
# Output: dictionary of arguments
# Parses the arguments
def parse_arguments(args):
    arg_parser = binary_search_parser()
    args, _ = arg_parser.parse_known_args(args)
    args = vars(args)
    return args


"""
@param arr1 : Input array
@param n    : Target element
@param None
linear_search in an array
"""
def linear_search(arr1,n):
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

if __name__ == "__main__":
    arr1 = [34,23,5,6,7,11,2,23,8,94,40,61,23,5]
    print(arr1)
    n = int(input("enter n value: "))
    search(arr1, n)
