import torch
from torch.autograd import Variable

def anneal_dsm_score_estimation(scorenet, samples, sigmas, labels=None, anneal_power=2., hook=None):
    if labels is None:
        labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = (sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:]))))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    target = - 1 / (used_sigmas ** 2) * noise
    scores = scorenet(perturbed_samples) #labels
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)

def batch_ger(s1_model, s1_model1, device):
    ger_results=torch.zeros([s1_model.shape[0], s1_model.shape[1], s1_model.shape[1]]).to(device)
    for i in range(len(s1_model1)):
        ger_results[i] = torch.ger(s1_model[i], s1_model1[i])
    # print(ger_results)
    return ger_results#d


def anneal_dsm_sec_score_estimation(scorenet, samples, sigmas, labels=None, anneal_power=2., hook=None):
    if labels is None:
        labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * (len(samples.shape[1:]))))
    # print("used_sigmas,", used_sigmas) #[batch_size, 1]
    noise = (torch.randn_like(samples) * used_sigmas).to(samples.device)
    perturbed_samples = samples + noise
    # print("used_sigmas", used_sigmas)
    # print("noise,", noise.shape)
    batch_eye = torch.zeros(samples.shape[0], samples.shape[1], samples.shape[1]).to(samples.device)
    batch_eye[:] = torch.eye(samples.shape[1]).to(samples.device)
    # print("batch_eye", batch_eye.shape)
    # print("noise,", noise.shape)
    target = 1 / (used_sigmas ** 2) * (1 / (used_sigmas.unsqueeze(dim=1) ** 2) * torch.bmm(noise.unsqueeze(dim=-1), noise.unsqueeze(dim=1))
                                       - batch_eye).view(samples.shape[0], samples.shape[1]**2) #batch_ger(noise, noise, noise.device
    scores = scorenet(perturbed_samples) #labels
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    # print("target,", target.shape)
    # print("scores,", scores.shape)

    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)

def joint_loss(scorenet2, scorenet1, samples, sigmas, labels=None, anneal_power=2., hook=None):
    if labels is None:
        labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * (len(samples.shape[1:]))))
    # print("used_sigmas,", used_sigmas) #[batch_size, 1]
    noise = (torch.randn_like(samples) * used_sigmas).to(samples.device)
    perturbed_samples = samples + noise
    perturbed_samples.requires_grad = True
    # print("used_sigmas", used_sigmas)
    # print("noise,", noise.shape)
    batch_eye = torch.zeros(samples.shape[0], samples.shape[1], samples.shape[1]).to(samples.device)
    batch_eye[:] = torch.eye(samples.shape[1]).to(samples.device)
    # print("batch_eye", batch_eye.shape)
    # print("noise,", noise.shape)
    # target = 1 / (used_sigmas ** 2) * (
    #             1 / (used_sigmas.unsqueeze(dim=1) ** 2) * torch.bmm(noise.unsqueeze(dim=-1), noise.unsqueeze(dim=1))
    #             - batch_eye).view(samples.shape[0], samples.shape[1] ** 2)  # batch_ger(noise, noise, noise.device
    score1 = scorenet1(perturbed_samples)# [samples, d]
    # print("score1.shape", score1.shape)

    #autograd.grad(torch.sum(s2_model, dim=0)[i, j], x_temp, create_graph=True)[0]
    g_s1 = torch.zeros([samples.shape[0], samples.shape[1], samples.shape[1]]).to(samples.device)
    for i in range(samples.shape[1]):
        g_s1[:, i, :] = torch.autograd.grad(torch.sum(score1, dim=0)[i], perturbed_samples, create_graph=True)[0]
    target = torch.bmm(score1.unsqueeze(-1), score1.unsqueeze(1)) + g_s1
    scores2 = scorenet2(perturbed_samples)  # labels [batch]

    loss = 1 / 2. * ((scores2 - target.view(samples.shape[0], -1)) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)