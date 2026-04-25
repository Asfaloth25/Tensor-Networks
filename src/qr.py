import torch
torch.set_default_dtype(torch.float64)

def qr_factorize(X:torch.Tensor, eps:float=1e-9)->tuple[torch.Tensor, torch.Tensor]:
    '''
    Vectorized QR matrix factorization using Gram-Schmidt algorithm.
    '''
    _, w = X.shape
    Q = torch.zeros_like(X)
    R = torch.zeros((w, w), device=X.device)

    for idx_col in range(w):
        col = X[:, idx_col].clone() # [h]

        prev_cols = Q[:, :idx_col] # [h, N]
        
        dot_prods = (
            prev_cols * col.unsqueeze(-1) # [h, N] * [h, 1] = [h, N]
        ).sum(dim=0) # [N]

        col -= (
            dot_prods * prev_cols # [N] * [h, N] = [h, N]
        ).sum(dim=-1) # [h]

        norm = torch.sqrt((col **2).sum()).clamp_min(eps)
        Q[:, idx_col] = col / norm

        R[:idx_col, idx_col] = dot_prods
        R[idx_col, idx_col] = norm

    return Q, R

def qr_factorize_tens(X:torch.Tensor, eps:float=1e-9)->tuple[torch.Tensor, torch.Tensor]:
    '''
    Vectorized QR factorization of the last 2 axes of `X` using Gram-Schmidt algorithm.
    '''
    _, w = X.shape[-2:]
    Q = torch.zeros_like(X)
    R = torch.zeros((*X.shape[:-2], w, w), device=X.device)

    for idx_col in range(w):
        col = X[..., idx_col].clone() # [..., h]

        prev_cols = Q[..., :idx_col] # [..., h, N]
        
        dot_prods = (
            prev_cols * col.unsqueeze(-1) # [..., h, N] * [..., h, 1] = [..., h, N]
        ).sum(dim=-2) # [..., N]

        col -= (
            dot_prods.unsqueeze(-2) * prev_cols # [..., N] * [..., h, N] = [..., h, N]
        ).sum(dim=-1) # [..., h]

        norm = torch.sqrt((col **2).sum(dim=-1)).clamp_min(eps) # [...]
        Q[..., idx_col] = col / norm.unsqueeze(-1)

        R[..., :idx_col, idx_col] = dot_prods
        R[..., idx_col, idx_col] = norm

    return Q, R


permutation_map = { # pre-QR, post-QR (inverse)
    'up':       ((1, 2, 0), (2, 0, 1)),
    'left':     ((0, 2, 1), (0, 2, 1)),
    'right':    ((0, 1, 2), (0, 1, 2))
}

def directional_node_qr(node:torch.Tensor, direction:str='up')->tuple[torch.Tensor, torch.Tensor]:
    '''
    direction must be in {"up", "left", "right"}

    T : [bond_dim, left, right]

    Q: [bond_dim, left, right] (isometric towards `direction`)

    R: {"up": [bond_dim, bond_dim], "left": [left, left], "right": [right, right]}
    '''

    direction = direction.lower()
    assert direction in permutation_map, f'Directional node QR does not support direction "{direction}".'
    
    dims = torch.tensor(node.shape)
    permutation = permutation_map[direction]

    node_reshaped = node.clone().permute(permutation[0]).flatten(start_dim=0, end_dim=1)
    Q, R = qr_factorize(node_reshaped)

    Q = Q.reshape([dims[i].item() for i in permutation[0]]).permute(permutation[1])
    return Q, R

einsum_map = {
    'up':       'b k, k i j -> b i j',
    'left':     'i k, b k j -> b i j',
    'right':    'j k, b i k -> b i j'
}
def absorb_r_node(node:torch.Tensor, R:torch.Tensor, direction:str='up'):
    '''
    `direction`: where the R comes from relative to `node`. For example, if `node` is
    the right child of "parent" and "parent" calculated R with respect to its right,
    `node` has to absorb R from direction `"up"`.
    '''

    assert direction in einsum_map, f'Directional R absorb does not support direction "{direction}".'

    return torch.einsum(einsum_map[direction], R, node)

if __name__ == "__main__":

    # seed = 2026; torch.manual_seed(seed)

    A = torch.rand((4, 4))
    Q, R = qr_factorize(A)


    print('TEST 1: Original matrix (A):\n', A)

    print('TESt 2: Q matrix:\n', Q)

    print('TEST 3: R matrix:\n', R)

    print('TEST 4: Modulus of Q columns:\n', (Q**2).sum(dim=0).sqrt())

    print('TEST 5: Q @ Q.T:\n', Q @ Q.T)

    print('TEST 6: (Q @ R) - A (six decimal places):\n', (Q @ R - A).round(decimals=6))

    B, C, D = (torch.rand((4, 4)) for _ in range(3))
    results = [qr_factorize(tens) for tens in (B, C, D)]
    X = torch.stack((B, C, D))
    Q_prima, R_prima = qr_factorize_tens(X)
    print(
        'TEST 7: Tensor QR diff - "qr_factorize" vs "qr_factorize_tens" (by layer):\n',
        [
            {
                "Q_diff_accumulated": (results[i][0]-Q_prima[i]).abs().sum(),
                "R_diff_accumulated": (results[i][1]-R_prima[i]).abs().sum()
            } for i in range(X.shape[0])
        ]
    )