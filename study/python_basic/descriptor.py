"""
Python Descriptor
    __get__()
    __set__()
    __delete()

    Object.__dict__['properties name']
    type(object).__dict__['properties name']
"""


class ReavealAccess(object):
    def __init__(self, initval=None, name="var"):
        self.val = initval
        self.name = name

    def __get__(self, obj, objtype):
        print("__get__", self.name)
        return self.val

    def __set__(self, obj, val):
        print("__set__", self.name)
        self.val = val


class MyClass(object):
    x = ReavealAccess(10, 'var "x"')


m = MyClass()
print(m.x)
print()
print(type(m).__dict__["x"].__get__(m, type(m)))
print()

m.x = 10
