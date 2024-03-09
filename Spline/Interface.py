import ctypes

lib = ctypes.CDLL('./lib/bin/spline.so')
lib.get_approximation.restype = ctypes.POINTER(ctypes.c_double)
lib.get_approximation.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_uint32]

def setSpline(a: float, b: float, n: int, fun_num: int, mu1: float = 0, mu2: float = 0):
    lib.set_spline(a, b, n, fun_num, mu1, mu2)

def getApproximation(a: float, b: float, n: int):
    ptr = lib.get_approximation(a, b, n)
    return ptr[:n]

setSpline(0, 1, 10, 1)
a = getApproximation(0, 1, 10)
print(a)



