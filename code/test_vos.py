import os
import time
import numpy as np

import torch

from data import vos
import utils
import utils.test_utils as test_utils
from models.encoder import make_encoder


def main(args):
    # building network
    if args.semantic_model is not None:
        semantic_model, mapScale = make_encoder(model_type=args.semantic_model,
                                                remove_layers=args.remove_layers)
        semantic_model.to(args.device)
        semantic_model.eval()
    else:
        semantic_model = None

    if args.fc_model is not None:
        fc_model, mapScale = make_encoder(model_type=args.fc_model,
                                          remove_layers=args.remove_layers)
        fc_model.to(args.device)
        fc_model.eval()
    else:
        fc_model = None

    args.mapScale = [8, 8]
    dataset = vos.VOSDataset(args)
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    late_fusion = semantic_model is not None and fc_model is not None

    with torch.no_grad():
        test(val_loader, semantic_model, fc_model, late_fusion, args)


def test(loader, semantic_model, fc_model, late_fusion, args):
    n_context = args.videoLen

    for vid_idx, (imgs, imgs_orig, lbls, lbls_orig, lbl_map, meta) in enumerate(loader):
        t_vid = time.time()
        imgs = imgs.to(args.device)
        B, N = imgs.shape[:2]
        assert (B == 1)

        print('******* Vid %s (%s frames) *******' % (vid_idx, N))
        with torch.no_grad():
            t00 = time.time()

            ##################################################################
            # Compute image features (batched for memory efficiency)
            ##################################################################
            bsize = 5  # minibatch size for computing features

            if semantic_model is not None:
                se_feats = []
                for b in range(0, imgs.shape[1], bsize):
                    feat = semantic_model(imgs[:, b:b + bsize].transpose(1, 2).to(args.device))
                    se_feats.append(feat.detach().cpu())
                se_feats = torch.cat(se_feats, dim=2).squeeze(1)
                se_feats = torch.nn.functional.normalize(se_feats, dim=1)

            if fc_model is not None:
                fc_feats = []
                for b in range(0, imgs.shape[1], bsize):
                    feat = fc_model(imgs[:, b:b + bsize].transpose(1, 2).to(args.device))
                    fc_feats.append(feat.detach().cpu())
                fc_feats = torch.cat(fc_feats, dim=2).squeeze(1)
                fc_feats = torch.nn.functional.normalize(fc_feats, dim=1)

            if late_fusion:
                feats = torch.cat((fc_feats * args.lambd, se_feats), dim=1)
                feats = torch.nn.functional.normalize(feats, dim=1)
            else:
                if semantic_model is not None:
                    feats = se_feats
                else:
                    feats = fc_feats
            print('computed features', time.time() - t00)
            ##################################################################
            # Compute affinities
            ##################################################################
            torch.cuda.empty_cache()
            t03 = time.time()

            # Prepare source (keys) and target (query) frame features
            key_indices = test_utils.context_index_bank(n_context, args.long_mem, N - n_context)
            key_indices = torch.cat(key_indices, dim=-1)
            keys, query = feats[:, :, key_indices], feats[:, :, n_context:]

            # Make spatial radius mask TODO use torch.sparse
            restrict = utils.MaskedAttention(args.radius, flat=False)
            D = restrict.mask(*feats.shape[-2:])[None]
            D = D.flatten(-4, -3).flatten(-2)
            D[D == 0] = -1e10
            D[D == 1] = 0

            # Flatten source frame features to make context feature set
            keys, query = keys.flatten(-2), query.flatten(-2)

            print('computing affinity')
            Ws, Is = test_utils.mem_efficient_batched_affinity(query, keys, D,
                                                               args.temperature, args.topk, args.long_mem, args.device)

            if torch.cuda.is_available():
                print(time.time() - t03, 'affinity forward, max mem', torch.cuda.max_memory_allocated() / (1024 ** 2))

            ##################################################################
            # Propagate Labels and Save Predictions
            ###################################################################

            maps, keypts = [], []
            lbls[0, n_context:] *= 0
            lbl_map, lbls = lbl_map[0], lbls[0]

            for t in range(key_indices.shape[0]):
                # Soft labels of source nodes
                ctx_lbls = lbls[key_indices[t]].to(args.device)
                ctx_lbls = ctx_lbls.flatten(0, 2).transpose(0, 1)

                # Weighted sum of top-k neighbours (Is is index, Ws is weight)
                pred = (ctx_lbls[:, Is[t]] * Ws[t].to(args.device)[None]).sum(1)
                pred = pred.view(-1, *feats.shape[-2:])
                pred = pred.permute(1, 2, 0)

                if t > 0:
                    lbls[t + n_context] = pred
                else:
                    pred = lbls[0]
                    lbls[t + n_context] = pred

                if args.norm_mask:
                    pred[:, :, :] -= pred.min(-1)[0][:, :, None]
                    pred[:, :, :] /= pred.max(-1)[0][:, :, None]

                # Save Predictions
                cur_img = imgs_orig[0, t + n_context].permute(1, 2, 0).numpy() * 255
                _maps = []

                outpath = os.path.join(args.save_path, str(vid_idx) + '_' + str(t))

                heatmap, lblmap, heatmap_prob = test_utils.dump_predictions(
                    pred.cpu().numpy(),
                    lbl_map, cur_img, outpath)

                _maps += [heatmap, lblmap, heatmap_prob]
                maps.append(_maps)

            if len(keypts) > 0:
                coordpath = os.path.join(args.save_path, str(vid_idx) + '.dat')
                np.stack(keypts, axis=-1).dump(coordpath)

            torch.cuda.empty_cache()
            print('******* Vid %s TOOK %s *******' % (vid_idx, time.time() - t_vid))


if __name__ == '__main__':
    args = utils.arguments.test_args()

    args.imgSize = args.cropSize
    print('Context Length:', args.videoLen, 'Image Size:', args.imgSize)
    print('Arguments', args)
    main(args)