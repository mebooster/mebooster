from __future__ import division
import torch
import torch.nn as nn
import torch.autograd as autograd


class Jacobian(nn.Module):
    '''
    Loss criterion that computes the trace of the square of the Jacobian.
    Arguments:
        n (int, optional): determines the number of random projections.
            If n=-1, then it is set to the dimension of the output
            space and projection is non-random and orthonormal, yielding
            the exact result.  For any reasonable batch size, the default
            (n=1) should be sufficient.
    '''

    def __init__(self, n=-1):
        assert n == -1 or n > 0
        self.n = n
        super(Jacobian, self).__init__()

    def forward(self, x, y):
        '''
        computes (1/2) tr |dy/dx|^2
        '''
        B, C = y.shape #32 10
        if self.n == -1:
            num_proj = C
        else:
            num_proj = self.n
        J2 = 0
        for ii in range(num_proj):
            if self.n == -1:
                # orthonormal vector, sequentially spanned
                v = torch.zeros(B, C)
                v[:, ii] = 1
            else:
                # random properly-normalized vector for each sample
                v = self._random_vector(C=C, B=B)
            if x.is_cuda:
                v = v.cuda()
            print("v:", v)
            Jv = self._jacobian_vector_product(y, x, v, create_graph=True)
            print("Jv:", Jv.shape)
            J2 += C * torch.norm(Jv) ** 2 / (num_proj * B)
        R = (1 / 2) * J2
        return R

    def _random_vector(self, C, B):
        '''
        creates a random vector of dimension C with a norm of C^(1/2)
        (as needed for the projection formula to work)
        '''
        if C == 1:
            return torch.ones(B)
        v = torch.randn(B, C)
        arxilirary_zero = torch.zeros(B, C)
        vnorm = torch.norm(v, 2, 1, True)
        v = torch.addcdiv(arxilirary_zero, 1.0, v, vnorm)
        return v

    def _jacobian_vector_product(self, y, x, v, create_graph=False):
        '''
        Produce jacobian-vector product dy/dx dot v.
        Note that if you want to differentiate it,
        you need to make create_graph=True
        '''
        flat_y = y.reshape(-1)
        flat_v = v.reshape(-1)
        grad_x, = torch.autograd.grad(flat_y, x, flat_v,
                                      retain_graph=True,
                                      create_graph=create_graph)
        return grad_x

class JacobianAugmentation(object):
    def __init__(self, copy_model):
        self.lammda=0.1
        self.copy_model=copy_model
        # self.queried_y=queried_y #should be torch

    def get_synthesizing_set(self, inputs, targets):
        #let input_shape a list
        # extended for Jacobian calculation


        # Jacobian computed by the improved method
        # On Colab CPU 0.16s, K80 GPU 0.14s
        # with JacobianMode(self.copy_model):
        #     out = self.copy_model(inputs).cpu()
        #     out.sum().backward()
        #     jac = self.copy_model.jacobian()
        inputs = inputs.cuda()
        targets = targets.cuda()
        inputs.requires_grad_()
        targets.requires_grad_()

        inputs.requires_grad = True
        outputs = self.copy_model(inputs)
        # pred = jacobian(targets, inputs, outputs, create_graph=True)#targets
        # jacobian=Jacobian()
        # jac_on_copy=jacobian(inputs, outputs)
        # print("jac_on_copy", jac_on_copy)
        grad_x, = torch.autograd.grad(outputs, inputs, targets, allow_unused=True)#retain_graph=True,
                                       #create_graph=True
        #outputs: _TensorOrTensors,
        #inputs: _TensorOrTensors,
        #grad_outputs
        #jacobian is calculated on the copy model/substitute model
        # print("grad_x", grad_x[0])
        # print('x', inputs[0])
        #calculate new set
        synthesizing_set = inputs + self.lammda * torch.sign(grad_x) #if all these two is torch
        # synthesizing_set=[]
        return synthesizing_set
