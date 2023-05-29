import torch
import numpy as np


'''
Utility functions for evaluation. 
'''


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def knn_classifier(
        train_features, train_labels, 
        test_features, test_labels, 
        k, T, 
        offset=0, # if the train_features are the same as the test features, offset should be set to 1
        num_classes=1000,
    ):

    # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, float, int, int) -> Tuple[float, float, int]

    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()

    ###########################################################
    # The indices of the topk neighbors for each image
    nn_inds = []
    ###########################################################
    
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)

        ###########################################################
        # collect the indices of the topk neighbors
        distances, indices = similarity.topk(k+offset, largest=True, sorted=True)
        distances, indices = distances[:,offset:], indices[:,offset:]
        nn_inds.append(indices.cpu().data.numpy())
        ###########################################################

        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, 5).sum().item()
        total += targets.size(0)
    
    nn_inds = np.concatenate(nn_inds, 0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5, nn_inds