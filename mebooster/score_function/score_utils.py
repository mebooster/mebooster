import torch
import torch.autograd as autograd
import numpy as np

def single_sliced_score_matching(energy_net, samples, noise=None, detach=False, noise_type='radermacher'):
    samples.requires_grad_(True)
    if noise is None:
        vectors = torch.randn_like(samples)
        if noise_type == 'radermacher':
            vectors = vectors.sign()
        elif noise_type == 'sphere':
            vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True) * np.sqrt(vectors.shape[-1])
        elif noise_type == 'gaussian':
            pass
        else:
            raise ValueError("Noise type not implemented")
    else:
        vectors = noise

    logp = -energy_net(samples).sum()
    grad1 = autograd.grad(logp, samples, create_graph=True)[0]
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(grad1 * vectors, dim=-1) ** 2 * 0.5
    if detach:
        loss1 = loss1.detach()
    grad2 = autograd.grad(gradv, samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)
    if detach:
        loss2 = loss2.detach()

    loss = (loss1 + loss2).mean()
    return loss, grad1, grad2

def sliced_score_estimation(score_net, samples, n_particles=1):
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)
    vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)

    grad1 = score_net(dup_samples)
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(grad1 * vectors, dim=-1) ** 2 * 0.5
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    return loss.mean(), loss1.mean(), loss2.mean()

def sliced_score_estimation_vr_fir_test(score_net, dup_samples, n_particles=1):
    # dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    # print(dup_samples.shape)
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)

    grad1 = score_net(dup_samples)
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    return loss.mean(), loss1.mean(), loss2.mean()

def sliced_score_estimation_vr_fir(score_net, samples, n_particles=1):
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    # print(dup_samples.shape)
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)

    grad1 = score_net(dup_samples)
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)
    loss = loss1 + loss2

    return loss.mean()#, loss1.mean(), loss2.mean()

def sliced_score_estimation_fir(score_net, dup_samples, n_particles=1):
    # dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    # print(dup_samples.shape)
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)
    vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)

    grad1 = score_net(dup_samples)
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(torch.pow(vectors * grad1, 2), dim=-1) / 2.
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    return loss.mean()#, loss1.mean(), loss2.mean()

# def sliced_score_estimation_vr_fir(score_net, samples, n_particles=1):
#     dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
#     dup_samples.requires_grad_(True)
#     vectors = torch.randn_like(dup_samples).to("cuda")
#
#     batch_size = dup_samples.shape[0]
#     d = dup_samples.shape[1]
#
#     grad1 = score_net(dup_samples)  # [n_samples, d]
#     # gradv = torch.sum(grad1 * vectors)
#     loss1 = torch.sum(grad1 * grad1, dim=-1) / 2  # [batch_size]
#     # grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
#     # print("loss1,shape", loss1.shape)
#     grad2 = torch.zeros([batch_size, d]).to("cuda")
#     grad1_sum = torch.sum(grad1, dim=0)
#     for i in range(grad1.shape[1]):
#         grad2[:, i] = autograd.grad(grad1_sum[i], dup_samples, create_graph=True)[0][:, i]
#
#     loss2 = torch.sum(grad2, dim=-1)
#     # print("loss2.shape", loss2.shape)
#
#     loss = loss1 + loss2
#     return loss.mean(), loss1.mean(), loss2.mean()

def sliced_score_estimation_vr_sec(score_net, samples, n_particles=1):
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    n = dup_samples.shape[-1]
    batch_size = dup_samples.shape[0]
    #1
    vectors = torch.randn(batch_size, n**2).to(dup_samples.device)
    grad1 = score_net(dup_samples)
    gradv = torch.sum(grad1 * vectors)#how to calculate batch?
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
    grad2_0 = autograd.grad(gradv, dup_samples, create_graph=True)[0] #batch_size \times n
    loss2_1 = torch.sum(grad2_0, dim=0)
    grad2_1 = torch.zeros(batch_size, n, n).to(dup_samples.device)
    for i in range(len(loss2_1)):
        grad2_1[:, :, i] = autograd.grad(loss2_1[i], dup_samples, create_graph=True)[0] #batch_size \tiems n
    loss2 = torch.sum(vectors * (grad2_1.view(batch_size, -1)), dim=-1)
    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)
    loss = loss1 - loss2
    #2
    # vectors = torch.randn(batch_size, n ** 2).to(dup_samples.device)
    # grad1 = score_net(dup_samples)
    # gradv = torch.sum(grad1 * vectors)  # how to calculate batch?
    # loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
    # grad2_0 = autograd.grad(gradv, dup_samples, create_graph=True)[0]  # batch_size \times n
    # loss2_1 = torch.sum(grad2_0, dim=0)
    # grad2_1 = torch.zeros(batch_size, n, n).to(dup_samples.device)
    # for i in range(len(loss2_1)):
    #     grad2_1[:, :, i] = autograd.grad(loss2_1[i], dup_samples, create_graph=True)[0]  # batch_size \tiems n
    # loss2 = torch.sum(vectors * (grad2_1.view(batch_size, -1)), dim=-1)
    # loss1 = loss1.view(n_particles, -1).mean(dim=0)
    # loss2 = loss2.view(n_particles, -1).mean(dim=0)
    # loss = loss + loss1 - loss2

    return loss.mean()#, loss1.mean(), loss2.mean()

def sliced_score_estimation_vr_thir(score_net, samples, n_particles=1):
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    n = dup_samples.shape[-1]
    batch_size = dup_samples.shape[0]
    # 1
    vectors = torch.randn(batch_size, n ** 3).to(dup_samples.device)
    grad1 = score_net(dup_samples)
    gradv = torch.sum(grad1 * vectors)  # how to calculate batch?
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
    grad2_0 = autograd.grad(gradv, dup_samples, create_graph=True)[0]  # batch_size \times n
    loss2_1 = torch.sum(grad2_0, dim=0)
    grad2_1 = torch.zeros(batch_size, n, n, n).to(dup_samples.device)
    for i in range(len(loss2_1)):
        step2 = autograd.grad(loss2_1[i], dup_samples, create_graph=True)[0]  # batch_size \tiems n
        for j in range(step2.shape[1]):
            grad2_1[:, i, j, :] = autograd.grad(torch.sum(step2[:, j]), dup_samples, create_graph=True)[0]

    loss2 = torch.sum(vectors * (grad2_1.view(batch_size, -1)), dim=-1)
    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)
    loss = loss1 + loss2
    # 2
    vectors = torch.randn(batch_size, n ** 3).to(dup_samples.device)
    grad1 = score_net(dup_samples)
    gradv = torch.sum(grad1 * vectors)  # how to calculate batch?
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
    grad2_0 = autograd.grad(gradv, dup_samples, create_graph=True)[0]  # batch_size \times n
    loss2_1 = torch.sum(grad2_0, dim=0)
    grad2_1 = torch.zeros(batch_size, n, n, n).to(dup_samples.device)
    # grad2_2 = torch.zeros(batch_size, n, n).to(dup_samples.device)
    for i in range(len(loss2_1)):
        step2 = autograd.grad(loss2_1[i], dup_samples, create_graph=True)[0]  # batch_size \tiems n
        for j in range(step2.shape[1]):
            grad2_1[:, i, j, :] = autograd.grad(torch.sum(step2[:, j]), dup_samples, create_graph=True)[0]

    loss2 = torch.sum(vectors * (grad2_1.view(batch_size, -1)), dim=-1)
    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)
    loss = loss + loss1 + loss2

    return loss.mean()


#denoised score estimation
