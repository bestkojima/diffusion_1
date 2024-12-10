class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

import torch
def svd_batch(matrix,k=None):
    U, S, V = torch.svd(matrix)
    
    
    "soft make"
    # a=torch.diag_embed(S)
    # temp=torch.zeros_like(a)
    # temp[:,:,:2,:2]=a[:,:,:2,:2]
    # a.shape
    reconstructed_matrix = torch.matmul(torch.matmul(U,torch.diag_embed(S)) , V.transpose(2,3))
    
    return reconstructed_matrix


if __name__ == '__main__':
    matrix = torch.randn(2,3,3,3)
    c=svd_batch(matrix)
    from torch.nn import functional as F
    z=F.mse_loss(c,matrix)
    print(z)