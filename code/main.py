import copy
import math
import os

import torch

import model as eg
from train import get_args, train_epoch, lambdas_from_args
from dataloader import build_train_val_dataset, make_train_loader
from evaluate import evaluate


def run_trial(args, data_loader, dataset, val_idx, device, radius, n_layers):
    """Train one model at a single (radius, n_layers) setting.

    Returns the best epoch's stats, with the best weights loaded back into the
    model before it is handed back.
    """
    torch.manual_seed(args.seed)   # same init/masking across trials -> fair comparison

    model = eg.EGNN(
        in_node_nf=args.in_node_nf,
        hidden_nf=args.hidden_nf,
        out_node_nf=args.out_node_nf,
        in_edge_nf=args.in_edge_nf,
        node_type_emb_nf=args.node_type_emb_nf,
        edge_type_emb_nf=args.edge_type_emb_nf,
        device=device,
        n_layers=n_layers,
        attention=args.attention,
        normalize=args.normalize,
        tanh=args.tanh,
        use_lm_emb=args.use_lm_emb,
        lm_emb_dim=args.lm_emb_dim,
        lm_proj_dim=args.lm_proj_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Per-step lr schedule: linear warmup over the first `warmup_frac` of all
    # optimizer steps, then cosine decay to 0 over the remainder. Stepped once
    # per optimizer step inside train_epoch (not per epoch).
    total_steps = max(1, len(data_loader) * args.epochs)
    warmup_steps = max(1, int(args.warmup_frac * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"[sched] {total_steps} total steps, {warmup_steps} warmup "
          f"({100 * warmup_steps / total_steps:.0f}%), then cosine to 0")

    schedule = args.mask_schedule
    best_gdt = -1.0        # checkpoint selection is on GDT-TS (higher is better)
    best_loss = float('inf')
    best_epoch = 0
    best_state = None
    best_stats = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        # Cycle the mask schedule over epochs: epoch1->10%, epoch2->20%, epoch3->30%, ...
        mask_fraction = schedule[(epoch - 1) % len(schedule)]
        stats = train_epoch(model, optimizer, data_loader, device, args, epoch,
                            mask_fraction=mask_fraction, scheduler=lr_scheduler)

        # Score on the held-out validation set: selection is on VALIDATION GDT-TS
        # (a generalisation estimate), not the training metric.
        val = evaluate(model, dataset, val_idx, device, args)
        val_gdt = val['mean'].get('gdt_ts', 0.0)

        comp = stats['components']
        comp_str = " ".join(f"{k}={v:.3f}" for k, v in comp.items() if k != 'total')
        print(
            f"epoch {epoch} => "
            f"train_loss {stats['loss']:.4f} | "
            f"masking {stats['masking_pct']:.1f}% | "
            f"train_GDT {stats['gdt_ts']:.4f} | "
            f"val_GDT {val_gdt:.4f}"
        )
        print(f"          components: {comp_str}")

        if val_gdt > best_gdt + args.min_delta:
            best_gdt = val_gdt
            best_loss = stats['loss']
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_stats = {**stats, 'val_gdt_ts': val_gdt, 'val_mean': val['mean']}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"[train] early stop at epoch {epoch}: no val GDT-TS improvement over "
                      f"{best_gdt:.4f} (epoch {best_epoch}) for {args.patience} epoch(s)")
                break

    ckpt_path = None
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[train] restored best weights from epoch {best_epoch} "
              f"(val GDT-TS {best_gdt:.4f}, train loss {best_loss:.4f})")

        os.makedirs(args.save_dir, exist_ok=True)
        ckpt_path = os.path.join(args.save_dir, f"r{radius:g}_l{n_layers}.pt")
        torch.save({
            'state_dict': best_state,
            'radius': radius,
            'n_layers': n_layers,
            'epoch': best_epoch,
            'train_loss': best_loss,
            'val_gdt_ts': best_gdt,
            'args': vars(args),
        }, ckpt_path)
        print(f"[train] saved {ckpt_path}")

    return {
        'radius': radius,
        'n_layers': n_layers,
        'best_epoch': best_epoch,
        'loss': best_loss,
        'val_gdt_ts': best_gdt,
        'train_gdt_ts': best_stats['gdt_ts'] if best_stats else 0.0,
        'checkpoint': ckpt_path,
        'model': model,
    }


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")

    radii = args.radius
    depths = args.n_layers
    print(f"[sweep] radius={radii} x n_layers={depths} => {len(radii) * len(depths)} run(s)")
    print(f"[loss]  lambdas: {lambdas_from_args(args)}")
    print(f"[train] {args.epochs} epoch(s) on {device}, mask schedule={args.mask_schedule}")
    if args.patience > 0:
        print(f"[train] early stopping: patience={args.patience} epoch(s), min_delta={args.min_delta}")

    results = []
    for radius in radii:
        # The radius is baked into the graph edges, so each radius needs its own
        # dataset build -- the expensive step. Hold it fixed and sweep depth inside.
        dataset, train_idx, val_idx = build_train_val_dataset(args, radius)
        data_loader = make_train_loader(dataset, train_idx,
                                        batch_size=args.batch_size, shuffle=True)
        train_nodes = sum(int(dataset[i]['x'].size(0)) for i in train_idx)
        print(f"[data] radius {radius}A: {len(train_idx)} train structure(s), "
              f"{train_nodes} nodes; {len(val_idx)} held out for validation")

        for n_layers in depths:
            print(f"\n=== trial: radius={radius}A, n_layers={n_layers} ===")
            results.append(run_trial(args, data_loader, dataset, val_idx,
                                     device, radius, n_layers))

    # Selection is on the VALIDATION GDT-TS -- a held-out generalisation estimate.
    # The final test set is supplied separately; run test.py on the winner for the
    # honest test-set number.
    results.sort(key=lambda r: r['val_gdt_ts'], reverse=True)
    print("\n=== sweep summary (sorted by validation GDT-TS) ===")
    print(f"{'radius':>7} {'layers':>7} {'epoch':>6} {'val_GDT':>8} {'train_GDT':>9} {'loss':>9}  checkpoint")
    for r in results:
        print(f"{r['radius']:>7g} {r['n_layers']:>7d} {r['best_epoch']:>6d} "
              f"{r['val_gdt_ts']:>8.4f} {r['train_gdt_ts']:>9.4f} {r['loss']:>9.4f}  {r['checkpoint']}")

    best = results[0]
    print(f"\n[sweep] best by validation GDT-TS: radius={best['radius']:g}A "
          f"n_layers={best['n_layers']} val_GDT={best['val_gdt_ts']:.4f} loss={best['loss']:.4f}")
    print(f"[next]  python test.py --checkpoint {best['checkpoint']} --write_cif")
    return best


if __name__ == "__main__":
    main()
