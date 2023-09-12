

def print_dict(dict_x, depth=0):
    for key, value in dict_x.items():
        if type(value) is dict:
            print_dict(value, depth + 1)
        elif type(value) is list:
            for i in value:
                print("[" + str(i) + "]")
        else:
            print("   " * depth, key + ":")
    print("-----------------------------------")
    for key, value in dict_x.items():
        if type(value) is dict:
            print_dict(value, depth + 1)
        elif type(value) is list:
            print("   " * depth, key + ":")
            for i in value:
                print("   " * depth, "[" + str(i) + "]")
                print("$$$")
        else:
            print("   " * depth, key + ":", value)
        print()