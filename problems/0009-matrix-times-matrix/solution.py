import torch

def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    a_t = torch.tensor(a,dtype=torch.float32)
    b_t = torch.tensor(b,dtype=torch.float32)
    
    # ensure that they are 2d and also 
    # the outer dim of inner maps to the outer 
    if (a_t.ndim != 2 or b_t.ndim != 2 or
        a_t.shape[1] != b_t.shape[0]):
        return torch.tensor(-1,dtype=torch.float32)
        #return torch.matmul(a_t,b_t)

    #return torch.tensor(-1,dtype=torch.float32)
    return a_t @ b_t