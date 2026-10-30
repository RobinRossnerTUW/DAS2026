import logging, os, argparse, math, random, re, glob
import pickle as pk
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch import optim
import torch.backends.cudnn as cudnn

from torchvision import transforms

from pytorch_metric_learning import samplers
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from utils.utils import GPU, seed_everything, load_config, getLogger, save_model, cosine_scheduler

from dataloading.writer_zoo import WriterZoo
from dataloading.GenericDataset import FilepathImageDataset
from dataloading.regex import pil_loader

from evaluators.retrieval import Retrieval
from page_encodings import SumPooling, GMP, MaxPooling, LSEPooling

from aug import Erosion, Dilation
from utils.triplet_loss import TripletLoss

from backbone import resnets
from backbone.model import Model

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def compute_page_features(_features, writer, pages):
    _labels = list(zip(writer, pages))

    labels_np = np.array(_labels)
    features_np = np.array(_features)
    writer = labels_np[:, 0]
    page = labels_np[:, 1]

    unique_pages = list(set(_labels))
    n_pages = len(unique_pages)
    logging.info(f'compute_page_features: {len(_features)} patches → {n_pages} pages')

    page_features = []
    page_writer = []
    for i, (w, p) in enumerate(tqdm(unique_pages, desc='Page Features')):
        idx = np.where((writer == w) & (page == p))
        page_features.append(features_np[idx])
        page_writer.append(w)

        if (i + 1) % 10 == 0 or (i + 1) == n_pages:
            logging.info(f'  Page Features: {i+1}/{n_pages} '
                         f'({100*(i+1)/n_pages:.1f}%)  '
                         f'last page had {len(features_np[idx])} patches')

    logging.info(f'compute_page_features done — {n_pages} page descriptors ready')
    return page_features, page_writer


# ── NEW: encode_per_class now optionally saves raw patch artifacts ─────────────
def encode_per_class(model, args, poolings=[], save_dir=None, split_name='test'):
    """
    Returns page-level pooled descriptors (unchanged behaviour).

    If save_dir is set, also writes to save_dir:
      raw_patches_{split_name}.npy        – (N_patches, D) patch embeddings before pooling
      patch_index_{split_name}.pkl        – list of dicts: {writer, page, patch_idx, filepath}
      patch_filepaths_{split_name}.txt    – one absolute filepath per line, same order as raw_patches
    """
    testset = args['testset']
    ds = WriterZoo.datasets[testset['dataset']]['set'][testset['set']]
    path = ds['path']
    regex = ds['regex']

    pfs_per_pooling = [[] for i in poolings]

    regex_w = regex.get('writer')
    regex_p = regex.get('page')

    srcs = sorted(list(glob.glob(f'{path}/**/*.png', recursive=True)))
    logging.info(f'Found {len(srcs)} images')
    writer = [int('_'.join(re.search(regex_w, Path(f).name).groups())) for f in srcs]
    page = [int('_'.join(re.search(regex_p, Path(f).name).groups())) for f in srcs]

    labels = list(zip(writer, page))

    if args.get('grayscale', None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3)
        ])
    else:
        transform = transforms.ToTensor()

    np_writer = np.array(writer)
    np_page = np.array(page)

    print(f'Found {len(list(set(labels)))} pages.')

    writers = []

    # Single HDF5 file for all patch embeddings -- 1 inode regardless of dataset size.
    # Each page is a dataset keyed by "writer_page" (e.g. "0042_3").
    # Random access: h5f["0042_3"][:]  -- no need to load the whole file.
    # Index pkl maps (writer, page) -> {hdf5_key, patch_idx_start, filepath list}.
    all_patch_index  = []
    all_filepaths    = []
    hdf5_file        = None
    global_patch_idx = 0

    if save_dir is not None:
        import h5py
        os.makedirs(save_dir, exist_ok=True)
        hdf5_path = os.path.join(save_dir, f'raw_patches_{split_name}.h5')
        hdf5_file = h5py.File(hdf5_path, 'w')

    for w, p in tqdm(set(labels), 'Page Features'):
        idx = np.where((np_writer == w) & (np_page == p))[0]
        fps = [srcs[i] for i in idx]
        page_ds = FilepathImageDataset(fps, pil_loader, transform)
        loader  = torch.utils.data.DataLoader(page_ds, num_workers=4, batch_size=args['test_batch_size'])

        feats = []
        for img in loader:
            img = img.cuda()
            with torch.no_grad():
                feat = model(img)
                feat = torch.nn.functional.normalize(feat)
            feats.append(feat.detach().cpu().numpy())

        feats = np.concatenate(feats)   # (n_patches_this_page, D)
        n     = feats.shape[0]

        # pooled descriptor (unchanged)
        for i, pooling in enumerate(poolings):
            enc = pooling.encode([feats])
            pfs_per_pooling[i].append(enc)
        writers.append(w)

        # write page patches into HDF5 -- O(page) memory, 1 inode total
        if hdf5_file is not None:
            key = f'{int(w):04d}_{int(p):04d}'
            hdf5_file.create_dataset(key, data=feats.astype(np.float32), compression='lzf')

            all_filepaths.extend(fps)
            all_patch_index.append({
                'writer':      int(w),
                'page':        int(p),
                'hdf5_key':    key,
                'patch_start': global_patch_idx,   # start row if loading all pages flat
                'patch_end':   global_patch_idx + n,
                'filepaths':   fps,
            })
            global_patch_idx += n

    torch.cuda.empty_cache()
    pfs_per_pooling = [np.concatenate(pfs) for pfs in pfs_per_pooling]

    # close HDF5 and write sidecar index
    if hdf5_file is not None:
        hdf5_file.close()
        print(f'Saved raw patches HDF5: {global_patch_idx} patches across {len(writers)} pages'
              f' -> raw_patches_{split_name}.h5')

        pk.dump(all_patch_index,
                open(os.path.join(save_dir, f'patch_index_{split_name}.pkl'), 'wb'))
        print(f'Saved patch index ({len(all_patch_index)} pages) -> patch_index_{split_name}.pkl')

        with open(os.path.join(save_dir, f'patch_filepaths_{split_name}.txt'), 'w') as f:
            f.write('\n'.join(all_filepaths))
        print(f'Saved filepaths ({len(all_filepaths)} lines) -> patch_filepaths_{split_name}.txt')

    return pfs_per_pooling, writers


def inference(model, ds, args):
    model.eval()
    loader = torch.utils.data.DataLoader(ds, num_workers=4, batch_size=args['test_batch_size'])

    feats = []
    pages = []
    writers = []

    for sample, labels in tqdm(loader, desc='Inference'):
        if len(labels) == 3:
            w, p = labels[1], labels[2]
        else:
            w, p = labels[0], labels[1]

        writers.append(w)
        pages.append(p)
        sample = sample.cuda()

        with torch.no_grad():
            emb = model(sample)
            emb = torch.nn.functional.normalize(emb)
        feats.append(emb.detach().cpu().numpy())

    feats   = np.concatenate(feats)
    writers = np.concatenate(writers)
    pages   = np.concatenate(pages)

    return feats, writers, pages


def test(model, logger, args, name='Test'):

    # define the poolings
    sumps, poolings = [], []
    sumps.append(SumPooling('l2', pn_alpha=0.4))
    poolings.append('SumPooling-PN0p4')

    # extract page descriptors + save raw patch artifacts for the test split
    pfs_per_pooling, writer = encode_per_class(
        model, args, poolings=sumps,
        save_dir=logger.log_dir,    # raw_patches_test.npy etc. land here
        split_name='test',
    )

    best_map    = -1
    best_top1   = -1
    best_pooling = ''
    best_pca     = None

    table   = []
    columns = ['Pooling', 'mAP', 'Top1']
    for i, pfs in enumerate(pfs_per_pooling):

        for pca_dim in [512]:

            if pca_dim != -1:
                pca_dim = min(min(pfs.shape), pca_dim)
                print(f'Fitting PCA with shape {pca_dim}')

                pca = PCA(pca_dim, whiten=True)
                pfs_tf = pca.fit_transform(pfs)
                pfs_tf = normalize(pfs_tf, axis=1)
            else:
                pca    = None
                pfs_tf = pfs

            print(f'Fitting PCA done')
            _eval = Retrieval()
            print(f'Calculate mAP..')

            res, _ = _eval.eval(pfs_tf, writer)

            p = f'{pca_dim}' if pca_dim != -1 else 'full'
            meanavp = res['map']

            np.save(os.path.join(logger.log_dir, f'embeddings_{poolings[i]}_{p}.npy'), pfs_tf)
            np.save(os.path.join(logger.log_dir, f'writers_{poolings[i]}_{p}.npy'), np.array(writer))

            if meanavp > best_map:
                best_map     = meanavp
                best_top1    = res['top1']
                best_pooling = f'{poolings[i]}-{p}'
                best_pca     = pca
                pk.dump(pca, open(os.path.join(logger.log_dir, 'pca.pkl'), 'wb'))

            table.append([f'{poolings[i]}-{p}', meanavp, res['top1']])
            print(f'{poolings[i]}-{p}-{name} MAP: {meanavp}')
            print(f'''{poolings[i]}-{p}-{name} Top-1: {res['top1']}''')

    logger.log_table(table, 'Results', columns)
    logger.log_value(f'Best-mAP',  best_map)
    logger.log_value(f'Best-Top1', best_top1)
    print(f'Best-Pooling: {best_pooling}')

    # ── Save machine-readable baseline metrics for compare_results.py ─────────
    import json as _json
    _res_full, _csv_res = _eval.eval(
        pfs_per_pooling[0] if len(pfs_per_pooling) == 1 else
        next(p for i,p in enumerate(pfs_per_pooling)),
        writer
    )
    # Re-evaluate the best pooling to get all metrics
    _best_idx = next(i for i, (pn, _) in enumerate(zip(poolings, sumps)) if f'{pn}-512' == best_pooling or pn in best_pooling)
    _pfs_best = pca.transform(pfs_per_pooling[_best_idx]) if best_pca else pfs_per_pooling[_best_idx]
    from sklearn.preprocessing import normalize as _norm
    _pfs_best = _norm(_pfs_best, axis=1)
    _, _metrics = Retrieval.calc_map_from_distances(
        np.array(writer), 1 - _pfs_best @ _pfs_best.T)
    _metrics_out = {
        'pooling': best_pooling,
        **{k: float(v) for k, v in _metrics.items()}
    }
    _json.dump(_metrics_out,
               open(os.path.join(logger.log_dir, 'metrics_baseline.json'), 'w'), indent=2)
    print(f'Saved metrics_baseline.json: mAP={_metrics_out["map"]:.4f}  Top1={_metrics_out["top1"]:.4f}')

    # return best PCA so main() can pass it to save_val_embeddings
    return best_pca


# ── NEW: save val embeddings projected with the same PCA as the test set ───────
def save_val_embeddings(model, val_ds, pca, log_dir, args):
    """
    Runs inference on val_ds, pools per page with SumPooling+PN,
    projects with the already-fitted test PCA, and saves:
      embeddings_val_SumPooling-PN0p4_512.npy
      writers_val_SumPooling-PN0p4_512.npy
      raw_patches_val.npy
      patch_index_val.pkl          (writer, page, patch_idx, filepath — best-effort)
    """
    print('=== Saving val embeddings ===')
    model.eval()

    # ── patch-level inference on val subset ───────────────────────────────────
    patch_feats, patch_writers, patch_pages = inference(model, val_ds, args)

    # ── save raw patch embeddings + index ─────────────────────────────────────
    np.save(os.path.join(log_dir, 'raw_patches_val.npy'), patch_feats)
    print(f'Saved raw val patches: {patch_feats.shape}')

    # build patch index (no filepaths available from the Subset loader, but
    # writer/page/patch_idx is enough to reconstruct everything)
    patch_index = [
        {'writer': int(w), 'page': int(p), 'patch_idx': i}
        for i, (w, p) in enumerate(zip(patch_writers, patch_pages))
    ]
    pk.dump(patch_index, open(os.path.join(log_dir, 'patch_index_val.pkl'), 'wb'))
    print(f'Saved patch index for val ({len(patch_index)} patches)')

    # ── pool per page ─────────────────────────────────────────────────────────
    page_feats, page_writers = compute_page_features(patch_feats, patch_writers, patch_pages)

    pooling = SumPooling('l2', pn_alpha=0.4)
    pfs = pooling.encode(page_feats)           # (N_pages, D) before PCA

    # save pre-PCA pooled descriptors too — useful for re-projecting later
    np.save(os.path.join(log_dir, 'embeddings_val_prepca.npy'), pfs)
    np.save(os.path.join(log_dir, 'writers_val_prepca.npy'), np.array(page_writers))
    print(f'Saved pre-PCA val pooled descriptors: {pfs.shape}')

    # ── project with the test PCA (same transform as test embeddings) ─────────
    if pca is not None:
        pfs_tf = pca.transform(pfs)
        pfs_tf = normalize(pfs_tf, axis=1)
    else:
        pfs_tf = pfs

    np.save(os.path.join(log_dir, 'embeddings_val_SumPooling-PN0p4_512.npy'), pfs_tf)
    np.save(os.path.join(log_dir, 'writers_val_SumPooling-PN0p4_512.npy'), np.array(page_writers))
    print(f'Saved val embeddings (PCA projected): {pfs_tf.shape}')

    # quick mAP on val so you can see it in the log
    _eval = Retrieval(options={'distance': 'cosine'})
    res, _ = _eval.eval(pfs_tf, np.array(page_writers))
    print(f'Val mAP (PCA projected, same as test): {res["map"]:.4f}  Top1: {res["top1"]:.4f}')

###########

def get_optimizer(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args['optimizer_options']['base_lr'],
                    weight_decay=args['optimizer_options']['wd'])
    return optimizer

def validate(model, val_ds, args):
    desc, writer, pages = inference(model, val_ds, args)
    print('Inference done')
    pfs, writer = compute_page_features(desc, writer, pages)

    norm    = 'powernorm'
    pooling = SumPooling(norm)
    descs   = pooling.encode(pfs)

    _eval = Retrieval()
    res, _ = _eval.eval(descs, writer)
    meanavp = res['map']

    return meanavp


def train_one_epoch(model, train_ds, triplet_loss, optimizer, scheduler, epoch, args, logger):

    model.train()
    model = model.cuda()

    sampler = samplers.MPerClassSampler(
        np.array(train_ds.dataset.labels[args['train_label']])[train_ds.indices],
        args['train_options']['sampler_m'],
        length_before_new_iter=args['train_options']['length_before_new_iter'])
    train_triplet_loader = torch.utils.data.DataLoader(
        train_ds, sampler=sampler,
        batch_size=args['train_options']['batch_size'],
        drop_last=True, num_workers=32)

    pbar = tqdm(train_triplet_loader)
    pbar.set_description('Epoch {} Training'.format(epoch))
    iters = len(train_triplet_loader)
    logger.log_value('Epoch', epoch, commit=False)

    for i, (samples, label) in enumerate(pbar):
        it = iters * epoch + i
        for i, param_group in enumerate(optimizer.param_groups):
            if it > (len(scheduler) - 1):
                param_group['lr'] = scheduler[-1]
            else:
                param_group['lr'] = scheduler[it]

            if param_group.get('name', None) == 'lambda':
                param_group['lr'] *= args['optimizer_options']['gmp_lr_factor']

        samples = samples.cuda()
        samples.requires_grad = True

        if args['train_label'] == 'cluster':
            l = label[0]
        elif args['train_label'] == 'writer':
            # CVL has no cluster label: (writer, page) → writer at [0]
            # ICDAR has cluster label:  (cluster, writer, page) → writer at [1]
            l = label[0] if len(label) == 2 else label[1]

        l = l.cuda()

        emb  = model(samples)
        loss = triplet_loss(emb, l, emb, l)
        logger.log_value(f'loss', loss.item())
        logger.log_value(f'lr', optimizer.param_groups[0]['lr'])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

    torch.cuda.empty_cache()
    return model

"""def train(model, train_ds, val_ds, args, logger, optimizer):

    epochs = args['train_options']['epochs']

    niter_per_ep = math.ceil(args['train_options']['length_before_new_iter'] / args['train_options']['batch_size'])
    lr_schedule  = cosine_scheduler(
        args['optimizer_options']['base_lr'], args['optimizer_options']['final_lr'],
        epochs, niter_per_ep,
        warmup_epochs=args['optimizer_options']['warmup_epochs'],
        start_warmup_value=0)

    best_epoch = -1
    best_map   = validate(model, val_ds, args)

    print(f'Val-mAP: {best_map}')
    logger.log_value('Val-mAP', best_map)

    loss = TripletLoss(margin=args['train_options']['margin'])
    print('Using Triplet Loss')

    for epoch in range(epochs):
        model = train_one_epoch(model, train_ds, loss, optimizer, lr_schedule, epoch, args, logger)
        mAP   = validate(model, val_ds, args)

        logger.log_value('Val-mAP', mAP)
        print(f'Val-mAP: {mAP}')

        if mAP > best_map:
            best_epoch = epoch
            best_map   = mAP
            save_model(model, optimizer, epoch, os.path.join(logger.log_dir, 'model.pt'))

        if (epoch - best_epoch) > args['train_options']['callback_patience']:
            break

    checkpoint = torch.load(os.path.join(logger.log_dir, 'model.pt'))
    print(f'''Loading model from Epoch {checkpoint['epoch']}''')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer"""

def train(model, train_ds, val_ds, args, logger, optimizer):

    epochs = args['train_options']['epochs']

    niter_per_ep = math.ceil(args['train_options']['length_before_new_iter'] / args['train_options']['batch_size'])
    lr_schedule  = cosine_scheduler(
        args['optimizer_options']['base_lr'], args['optimizer_options']['final_lr'],
        epochs, niter_per_ep,
        warmup_epochs=args['optimizer_options']['warmup_epochs'],
        start_warmup_value=0)

    loss = TripletLoss(margin=args['train_options']['margin'])
    print('Using Triplet Loss')

    for epoch in range(epochs):
        model = train_one_epoch(model, train_ds, loss, optimizer, lr_schedule, epoch, args, logger)
        save_model(model, optimizer, epoch, os.path.join(logger.log_dir, 'model.pt'))
        print(f'Epoch {epoch} done — model saved')

    # load final model
    checkpoint = torch.load(os.path.join(logger.log_dir, 'model.pt'))
    print(f'''Loading model from Epoch {checkpoint['epoch']}''')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

def prepare_logging(args):
    os.path.join(args['log_dir'], args['super_fancy_new_name'])
    Logger  = getLogger(args['logger'])
    logger  = Logger(os.path.join(args['log_dir'], args['super_fancy_new_name']), args=args)
    logger.log_options(args)
    return logger

def train_val_split(dataset, prop=0.9):
    authors = list(set(dataset.labels['writer']))
    random.shuffle(authors)

    train_len    = math.floor(len(authors) * prop)
    train_authors = authors[:train_len]
    val_authors   = authors[train_len:]

    print(f'{len(train_authors)} authors for training - {len(val_authors)} authors for validation')

    train_idxs = []
    val_idxs   = []

    for i in tqdm(range(len(dataset)), desc='Splitting dataset'):
        w = dataset.get_label(i)[1]
        if w in train_authors:
            train_idxs.append(i)
        if w in val_authors:
            val_idxs.append(i)

    train = torch.utils.data.Subset(dataset, train_idxs)
    val   = torch.utils.data.Subset(dataset, val_idxs)

    return train, val

def main(args):
    logger = prepare_logging(args)
    logger.update_config(args)

    backbone = getattr(resnets, args['model']['name'], None)()
    if not backbone:
        print("Unknown backbone!")
        raise

    print('----------')
    print(f'Using {type(backbone)} as backbone')
    print(f'''Using {args['model'].get('encoding', 'netvlad')} as encoding.''')
    print('----------')

    random_ = args['model'].get('encoding', None) == 'netrvlad'
    model   = Model(backbone, dim=64, num_clusters=args['model']['num_clusters'], random=random_)
    model.train()
    model = model.cuda()

    tfs = []

    if args.get('grayscale', None):
        tfs.extend([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3)
        ])
    else:
        tfs.append(transforms.ToTensor())

    if args.get('data_augmentation', None) == 'morph':
        tfs.extend([
            transforms.RandomApply([Erosion()],   p=0.3),
            transforms.RandomApply([Dilation()],  p=0.3),
        ])

    transform = transforms.Compose(tfs)

    train_dataset = None
    if args['trainset']:
        d = WriterZoo.get(**args['trainset'])
        # Derive label names from regex keys so datasets without 'cluster' (e.g. CVL) work automatically
        train_regex = WriterZoo.datasets[args['trainset']['dataset']]['set'][args['trainset']['set']]['regex']
        train_label_names = list(train_regex.keys())
        train_dataset = d.TransformImages(transform=transform).SelectLabels(label_names=train_label_names)

    if args.get('use_test_as_validation', False):
        val_ds = WriterZoo.get(**args['testset'])
        if args.get('grayscale', None):
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=3)
            ])
        else:
            test_transform = transforms.ToTensor()
        val_ds = val_ds.TransformImages(transform=test_transform).SelectLabels(label_names=['writer', 'page'])

        train_ds = torch.utils.data.Subset(train_dataset, range(len(train_dataset)))
        val_ds   = torch.utils.data.Subset(val_ds, range(len(val_ds)))
    else:
        # train_ds, val_ds = train_val_split(train_dataset)
        train_ds = torch.utils.data.Subset(train_dataset, range(len(train_dataset)))
        val_ds   = train_ds  # unused placeholder, train() no longer calls validate()
        print(f'Training on all {len(train_dataset)} patches from all writers')

    optimizer = get_optimizer(args, model)

    if args['checkpoint']:
        print(f'''Loading model from {args['checkpoint']}''')
        checkpoint = torch.load(args['checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if not args['only_test']:
        model, optimizer = train(model, train_ds, val_ds, args, logger, optimizer)

    # ── testing ───────────────────────────────────────────────────────────────
    save_model(model, optimizer, args['train_options']['epochs'], os.path.join(logger.log_dir, 'model.pt'))

    # test() now returns the fitted PCA
    best_pca = test(model, logger, args, name='Test')

    # ── NEW: save val embeddings projected with the same PCA ──────────────────
    # Skip when use_test_as_validation=True (val IS the test set already)
    if not args.get('use_test_as_validation', False):
        save_val_embeddings(model, val_ds, best_pca, logger.log_dir, args)

    logger.finish()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')
    parser = argparse.ArgumentParser()

    parser.add_argument('--config',    default='config/icdar2017.yml')
    parser.add_argument('--only_test', default=False, action='store_true',
                        help='only test')
    parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--no-cuda',   action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpuid',     default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed',      default=2174, type=int,
                        help='seed')

    args = parser.parse_args()

    config = load_config(args)[0]

    GPU.set(args.gpuid, 400)
    cudnn.benchmark = True

    seed_everything(args.seed)

    main(config)