import numpy as np
from matplotlib import cm
import cv2
import imageio

import torch
import torch.nn.functional as F


def dump_predictions(pred, lbl_set, img, prefix):
    '''
    Save:
        1. Predicted labels for evaluation
        2. Label heatmaps for visualization
    '''
    sz = img.shape[:-1]

    # Upsample predicted soft label maps
    # pred_dist = pred.copy()
    pred_dist = cv2.resize(pred, sz[::-1])[:]

    # Argmax to get the hard label for index
    pred_lbl = np.argmax(pred_dist, axis=-1)
    pred_lbl = np.array(lbl_set, dtype=np.int32)[pred_lbl]
    img_with_label = np.float32(img) * 0.5 + np.float32(pred_lbl) * 0.5

    # Visualize label distribution for object 1 (debugging/analysis)
    pred_soft = pred_dist[..., 1]
    pred_soft = cv2.resize(pred_soft, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred_soft = cm.jet(pred_soft)[..., :3] * 255.0
    img_with_heatmap1 = np.float32(img) * 0.5 + np.float32(pred_soft) * 0.5

    # Save blend image for visualization
    imageio.imwrite('%s_blend.jpg' % prefix, np.uint8(img_with_label))

    if prefix[-4] != '.':  # Super HACK-y
        imname2 = prefix + '_mask.png'
        # skimage.io.imsave(imname2, np.uint8(pred_val))
    else:
        imname2 = prefix.replace('jpg', 'png')
        # pred_val = np.uint8(pred_val)
        # skimage.io.imsave(imname2.replace('jpg','png'), pred_val)

    # Save predicted labels for evaluation
    imageio.imwrite(imname2, np.uint8(pred_lbl))

    return img_with_label, pred_lbl, img_with_heatmap1


def context_index_bank(n_context, long_mem, N):
    '''
    Construct bank of source frames indices, for each target frame
    '''
    ll = []   # "long term" context (i.e. first frame)
    for t in long_mem:
        assert 0 <= t < N, 'context frame out of bounds'
        idx = torch.zeros(N, 1).long()
        if t > 0:
            idx += t + (n_context+1)
            idx[:n_context+t+1] = 0
        ll.append(idx)
    # "short" context    
    ss = [(torch.arange(n_context)[None].repeat(N, 1) + torch.arange(N)[:, None])[:, :]]

    return ll + ss


def mem_efficient_batched_affinity(query, keys, mask, temperature, topk, long_mem, device):
    '''
    Mini-batched computation of affinity, for memory efficiency
    '''
    bsize, pbsize = 2, 100   # keys.shape[2] // 2
    Ws, Is = [], []

    for b in range(0, keys.shape[2], bsize):
        _k, _q = keys[:, :, b:b+bsize].to(device), query[:, :, b:b+bsize].to(device)
        w_s, i_s = [], []

        for pb in range(0, _k.shape[-1], pbsize):
            A = torch.einsum('ijklm,ijkn->iklmn', _k, _q[..., pb:pb+pbsize]) 
            A[0, :, len(long_mem):] += mask[..., pb:pb+pbsize].to(device)

            _, N, T, h1w1, hw = A.shape
            A = A.view(N, T*h1w1, hw)
            A /= temperature

            weights, ids = torch.topk(A, topk, dim=-2)
            weights = F.softmax(weights, dim=-2)
            
            w_s.append(weights.cpu())
            i_s.append(ids.cpu())

        weights = torch.cat(w_s, dim=-1)
        ids = torch.cat(i_s, dim=-1)
        Ws += [w for w in weights]
        Is += [ii for ii in ids]

    return Ws, Is


def batched_affinity(query, keys, mask, temperature, topk, long_mem, device):
    '''
    Mini-batched computation of affinity, for memory efficiency
    (less aggressively mini-batched)
    '''
    bsize = 2
    Ws, Is = [], []
    for b in range(0, keys.shape[2], bsize):
        _k, _q = keys[:, :, b:b+bsize].to(device), query[:, :, b:b+bsize].to(device)
        w_s, i_s = [], []

        A = torch.einsum('ijklmn,ijkop->iklmnop', _k, _q) / temperature
        
        # Mask
        A[0, :, len(long_mem):] += mask.to(device)

        _, N, T, h1w1, hw = A.shape
        A = A.view(N, T*h1w1, hw)
        A /= temperature

        weights, ids = torch.topk(A, topk, dim=-2)
        weights = F.softmax(weights, dim=-2)
            
        Ws += [w for w in weights]
        Is += [ii for ii in ids]

    return Ws, Is