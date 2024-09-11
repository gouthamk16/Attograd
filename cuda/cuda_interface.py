import ctypes
import numpy as np

lib = ctypes.CDLL('shared_lib/vector_ops.so')

lib.matMul.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                          np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                            np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]


class needCuda:
    def __init__(self):
        self.cuda = False
    

def matrixMultiply(A, B):
    m, n = A.shape
    j, k = B.shape
    if n!=j and k==m:
        # Swap two matrices
        A, B = B, A
        m, n = A.shape
        j, k = B.shape
    elif n!=j:
        raise ValueError("Matrix dimensions do not match")
    
    C = np.zeros((m, k), dtype=np.float32)

    lib.matMul(A, B, C, m, j, n, k)

    return C

# if __name__ == "__main__":
#     A = np.random.rand(100, 100).astype(np.float32)
#     B = np.random.rand(100, 100).astype(np.float32)

#     C = matrixMultiply(A, B)
#     print(C)