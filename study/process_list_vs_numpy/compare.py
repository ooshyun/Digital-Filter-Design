
"""
Compare difference of arr1,arr2 and offset

return
None: instance error, it's not the numpy array
1: Same arr1, arr2
2: Error of being different between two arrays on shape
3. Error of difference was bigger than offset
"""

import numpy as np


def compare_array(arr1, arr2, offset):
    if isinstance(arr1, np.ndarray) and isinstance(arr1, np.ndarray):
        if arr1.shape != arr2.shape:
            print(f"ERROR: array shape mismatch, {arr1.shape} {arr2.shape}")
            return 2

        arr1_np = np.array(arr1)
        arr2_np = np.array(arr2)
        # print(arr1_np[0], arr2_np[0])
        truth_table = abs(arr1_np-arr2_np) < np.ones(shape=arr1_np.shape)*offset

        if not truth_table.all():
            # print(abs(arr1_np-arr2_np))
            print("ERROR: error value is too large")
            return 2
        else:
            return 1
    else:
        compare_array(np.array(arr1), np.array(arr2), offset)


if __name__=='__main__':
    """
    For test verifying function
    """
    import numpy as np

    arr = np.array((1., 2., 3.), dtype='float64')
    arr2 = np.array((1., 2., 5.), dtype='float64')
    x = np.array((arr, arr, arr))
    y = np.array((arr2, arr2, arr2))

    print("-------------------Test1-------------------")
    # compare_array(x, y)
    comp1 = abs(np.matrix(np.matrix(x) - np.matrix(y)))
    comp2 = np.matrix(np.ones(shape=np.matrix(np.matrix(x)).shape))
    test1 = comp1 <= comp2
    print(f"Subtraction: \n"
        f"{comp1}")
    print(f"Offset: \n"
        f"{comp2}")
    print(f"Comparison: \n"
        f"{test1}")

    print("-------------------Test2-------------------")
    # compare_array(x, y) in simple version
    comp1 = abs(np.matrix(x) - np.matrix(y))
    comp2 = np.matrix(np.ones(shape=np.matrix(np.matrix(x) - np.matrix(y)).shape))
    test2 = comp1 <= comp2

    print(f"Subtraction: \n"
        f"{comp1}")
    print(f"Offset: \n"
        f"{comp2}")
    print(f"Comparison: \n"
        f"{test2}")


    print("-------------------Test3-------------------")
    # compare_array(x, y) in most simple version
    comp1 = abs(x-y)
    comp2 = np.ones(shape=x.shape)
    test3 = comp1 <= comp2

    print(f"Subtraction: \n"
        f"{comp1}")
    print(f"Offset: \n"
        f"{comp2}")
    print(f"Comparison: \n"
        f"{test3}")

    # Check All TRUE?
    print(f"ALL TRUE? {test3.all()}")

    # Check 1 TRUE?
    print(f"TRUE is EXISTED? {test3.any()}")

    # Check numpy array type
    print(f"instance is numpy? {isinstance(arr, np.ndarray)}")

