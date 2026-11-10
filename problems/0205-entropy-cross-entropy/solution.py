import torch

def entropy_and_cross_entropy(P: torch.Tensor, Q: torch.Tensor) -> tuple[float, float]:
    """
    Compute entropy of P and cross-entropy between P and Q.
    
    Args:
        P: True probability distribution (torch.Tensor)
        Q: Predicted probability distribution (torch.Tensor)
    
    Returns:
        Tuple of (entropy H(P), cross-entropy H(P,Q))
    """
    # Your code here

    # mathematically correct, but need to address 3 things:
    # NaN BUG 
    # Dimension / Batching Issues 
    # Type Hint Mismatch 
    # entropy_p = - torch.sum(P * torch.log(P))
    # cross_entropy_P_Q = - torch.sum(P * torch.log(Q))

    # return (entropy_p, cross_entropy_P_Q)

    eps = 1e-12

    log_P = torch.log(P.clamp(min=eps))
    log_Q = torch.log(Q.clamp(min=eps))

    # compute per sample metrics by summing over the class dimensions 
    entropy_per_sample = -torch.sum(P* log_P,dim=-1)
    cross_entropy_per_sample = -torch.sum(P*log_Q,dim=-1)

    # take the mean over the batch and convert to floats 
    mean_entropy = torch.mean(entropy_per_sample).item()
    mean_cross_entropy = torch.mean(cross_entropy_per_sample).item()

    return (mean_entropy,mean_cross_entropy)