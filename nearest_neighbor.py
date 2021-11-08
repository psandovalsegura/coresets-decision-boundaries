import torch
import pdb

def create_cifar_coreset_tensor(model, trainset):
    # CIFAR-10 coreset indices
    fcnn_sel = [13792,17049,27470,35798,37416,38974,40664,46805,46996,47577]

    coreset_raw = torch.utils.data.Subset(trainset, fcnn_sel)
    coresetloader = torch.utils.data.DataLoader(
        coreset_raw, batch_size=len(fcnn_sel), shuffle=False, num_workers=1)

    model.eval()
    with torch.no_grad():
        inputs, targets = iter(coresetloader).next()
        inputs, targets = inputs.cuda(), targets.cuda()
        coreset_matrix = model(inputs, last_layer=True)
        coreset_target = targets

    return coreset_matrix, coreset_target

class nnclass:
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.targets = target
    
    def classify(self, inputs):
        outputs = []
        for i in range(len(inputs)):
            x = inputs[i]
            dist = torch.norm(self.dataset - x, dim=1, p=2)
            nn = dist.topk(1, largest=False)
            outputs.append(self.targets[nn.indices.item()])
        return torch.stack(outputs)
