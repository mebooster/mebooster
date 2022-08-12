#active learning: prada : T-RND I-FGSM: target random iterative fast gradient sign method
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd

class NonTargetFGSM(object):
    def __init__(self, copy_model):
        self.epsilon=5/255 #64,255
        self.copy_model=copy_model
        # self.queried_y=queried_y

    def creterion(self, pred, soft_targets, weights=None):#cross entropy
        if weights is not None:
            # print("weights is not None")
            return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
        else:
            return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))

    def where(self, cond, x, y):
        """
        code from :
            https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
        """
        cond = cond.float()
        return (cond * x) + ((1 - cond) * y)

    def get_synthesizing_set(self, inputs, targets, epsilon=None):
        if epsilon is not None:
           self.epsilon=epsilon
        inputs = inputs.cuda()
        targets = targets.cuda()
        x_adv = inputs
        x_adv.requires_grad_()
        targets.requires_grad_()
        x_val_min = -1
        x_val_max = 1
        iteration=1
        targeted = False
        alpha=1
        for i in range(iteration):
            outputs = self.copy_model(x_adv)
            if targeted:
                cost = self.creterion(outputs, targets)
            else:
                cost = -self.creterion(outputs, targets)

            # print("x_adv.grad,", x_adv.grad)
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            self.copy_model.zero_grad()
            cost.backward()
            # print("x_adv.grad,", x_adv.grad)

            # x_adv.grad.sign_() #here we already got thier sign
            u = torch.sign(x_adv.grad.data)
            x_adv = x_adv - self.epsilon * u#torch.sign(x_adv.grad.data) #x_adv.grad.data
            print(x_adv.shape)
            # x_adv = self.where(x_adv > inputs + self.epsilon, inputs + self.epsilon, x_adv)
            # x_adv = self.where(x_adv < inputs - self.epsilon, inputs - self.epsilon, x_adv)
            # x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)

        return x_adv.detach()#, u.cpu()