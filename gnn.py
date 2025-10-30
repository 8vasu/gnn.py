#!/usr/bin/sage --python

# gnn.py - Geometric Graph Neural Networks.
# Copyright (C) 2025 Soumendra Ganguly

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, networkx as nx, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse, random, time, statistics, json, os
from datetime import datetime

# -------------------------
#  Hyperbolic Geometry Utils
# -------------------------
class H:
    def __init__(self, c=1.0):
        self.c, self.eps, self.r, self.cs = c, 1e-7, 1.0/math.sqrt(c), math.sqrt(c)

    def vnorm(self, x):
        return torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp(min=self.eps)

    def project(self, x):
        n = self.vnorm(x)
        m = (1-self.eps)*self.r
        return torch.where(n>m, x/n*m, x)

    def expmap0(self, v):
        vn = self.vnorm(v)
        sc = torch.where(vn<self.eps, self.r,
                         torch.tanh(self.cs*vn)/(self.cs*vn))
        return self.project(sc*v)

    def logmap0(self, x):
        xn = self.vnorm(x)
        at = torch.where(xn<self.eps, self.cs,
                         torch.atanh(torch.clamp(self.cs*xn, max=1-self.eps))/xn)
        return at*x

# -------------------------
#  Graph Data Generator
# -------------------------
def get_graph_data(n=2000, nc=10, nf=128, geom='euclidean'):
    G = nx.watts_strogatz_graph(n, k=6, p=0.2)
    A = torch.FloatTensor(nx.adjacency_matrix(G).todense() + np.eye(n))
    di = torch.pow(A.sum(1), -0.5)
    di[torch.isinf(di)] = 0
    A = di.unsqueeze(1) * A * di.unsqueeze(0)

    if geom == 'euclidean':
        centers = torch.randn(nc, 2) * 3
        xs, ys = [], []
        samples_per_class = n // nc
        remainder = n - samples_per_class * nc
        for i in range(nc):
            count = samples_per_class + (1 if i < remainder else 0)
            xs.append(centers[i] + 0.5 * torch.randn(count, 2))
            ys += [i] * count
        x = torch.cat(xs, dim=0)
        y = torch.tensor(ys[:n])
        W = torch.randn(x.shape[1], x.shape[1])
        Q, _ = torch.linalg.qr(W)
        x = x @ Q
        x = x - x.mean(0)
        W_expand = torch.randn(x.shape[1], nf) / math.sqrt(x.shape[1])
        x = x @ W_expand
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)

    elif geom == 'spherical':
        lat = torch.linspace(0, math.pi, n)
        lon = torch.linspace(0, 2 * math.pi, n)
        x = torch.stack([
            torch.sin(lat) * torch.cos(lon),
            torch.sin(lat) * torch.sin(lon),
            torch.cos(lat)
        ], dim=1)
        x = F.normalize(x + 0.05 * torch.randn_like(x))
        y = torch.floor((x[:, 2] + 1) / 2 * nc).clamp(0, nc - 1).long()
        W = torch.randn(x.shape[1], nf) / math.sqrt(x.shape[1])
        x = x @ W
        x = F.normalize(x)

    else:  # hyperbolic
        depths = dict(nx.shortest_path_length(G, 0))
        x = torch.zeros(n, 2)
        for i in range(n):
            d = depths.get(i, 0)
            angle = 2 * math.pi * (i % (2 ** min(d, 8))) / (2 ** min(d, 8))
            r = math.tanh(0.3 * d)
            x[i] = torch.tensor([r * math.cos(angle), r * math.sin(angle)])
        x += 0.05 * torch.randn_like(x)
        y = torch.tensor([min(depths.get(i, 0) * nc // 8, nc - 1) for i in range(n)])
        W = torch.randn(x.shape[1], nf)
        x = x @ W
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)

    idx = torch.randperm(n)
    tr = torch.zeros(n, dtype=torch.bool)
    vl = torch.zeros(n, dtype=torch.bool)
    te = torch.zeros(n, dtype=torch.bool)
    tr[idx[:int(0.6 * n)]] = True
    vl[idx[int(0.6 * n):int(0.8 * n)]] = True
    te[idx[int(0.8 * n):]] = True

    return x, A, y, tr, vl, te

# -------------------------
#  GNN Layers
# -------------------------
class GC(nn.Module):
    def __init__(self, in_d, out_d, geom, c=1.0):
        super().__init__()
        self.geom = geom
        self.H = H(c) if geom == 'hyperbolic' else None
        self.W = nn.Parameter(torch.randn(out_d, in_d))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, A):
        if self.geom == 'euclidean':
            x = torch.mm(A, x)
            x = F.linear(x, self.W)
            return x
        elif self.geom == 'spherical':
            x = F.normalize(x, dim=-1)
            x = torch.mm(A, x)
            x = F.linear(x, self.W)
            x = F.relu(x)
            return F.normalize(x, dim=-1)
        else:
            x = self.H.logmap0(self.H.project(x))
            x = F.linear(torch.mm(A, x), self.W)
            return self.H.expmap0(x)

class GNN(nn.Module):
    def __init__(self, in_d, h_d, out_d, geom, c=1.0):
        super().__init__()
        self.geom = geom
        self.H = H(c) if geom == 'hyperbolic' else None
        self.c1 = GC(in_d, h_d, geom, c)
        self.c2 = GC(h_d, out_d, geom, c)

    def forward(self, x, A):
        x = self.c1(x, A)
        x = F.dropout(F.relu(x), 0.3, self.training)
        x = self.c2(x, A)
        return F.log_softmax(x, dim=1)

# -------------------------
#  Visualization
# -------------------------
def visualize_graph(x, y, A, geom, path, verbose=True):
    """
    Visualize graph embeddings with PCA projection to 2D.
    
    Args:
        x: Node features/embeddings
        y: Node labels
        A: Adjacency matrix
        geom: Geometry type ('euclidean', 'spherical', 'hyperbolic')
        path: Save path for the image
        verbose: Whether to print save confirmation
    """
    try:
        from sklearn.decomposition import PCA
        x_np = x.detach().cpu().numpy()
        if x_np.shape[1] > 2:
            x_2d = PCA(n_components=2).fit_transform(x_np)
            x_vis = torch.tensor(x_2d)
        else:
            x_vis = x.detach().cpu()
    except ImportError:
        if verbose:
            print("[WARNING] scikit-learn not available, skipping visualization")
        return
    except Exception as e:
        if verbose:
            print(f"[WARNING] Visualization failed: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw edges
    A_cpu = A.cpu() if torch.is_tensor(A) else A
    ei = torch.nonzero(A_cpu > 0.01)
    for i, j in ei:
        if i < j:
            ax.plot([x_vis[i, 0], x_vis[j, 0]], 
                   [x_vis[i, 1], x_vis[j, 1]], 
                   'gray', alpha=0.3, linewidth=0.5)
    
    # Draw nodes by class
    y_cpu = y.cpu() if torch.is_tensor(y) else y
    for c in range(y_cpu.max().item() + 1):
        m = y_cpu == c
        ax.scatter(x_vis[m, 0], x_vis[m, 1], s=50, label=f'C{c}', alpha=0.7)
    
    # Special handling for hyperbolic geometry
    if geom == 'hyperbolic':
        circle = Circle((0, 0), 1, fill=False, linewidth=2, color='black')
        ax.add_patch(circle)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
    
    ax.set_title(f'{geom.capitalize()} GNN Embeddings')
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    
    try:
        plt.savefig(path, dpi=150)
        if verbose:
            print(f"[VIZ] Saved {path}")
    except Exception as e:
        if verbose:
            print(f"[WARNING] Failed to save visualization: {e}")
    finally:
        plt.close()

# -------------------------
#  Training
# -------------------------
def train(mdl, x, A, y, masks, epochs=500, verbose=True):
    tr, vl = masks
    opt = torch.optim.Adam(mdl.parameters(), lr=0.01, weight_decay=5e-4)
    best = 0
    for e in range(epochs):
        mdl.train()
        opt.zero_grad()
        out = mdl(x, A)
        loss = F.nll_loss(out[tr], y[tr])
        loss.backward()
        opt.step()
        if e % 50 == 0:
            mdl.eval()
            with torch.no_grad():
                out_eval = mdl(x, A)
                acc = (out_eval[vl].argmax(1) == y[vl]).float().mean().item()
            best = max(best, acc)
            if verbose:
                print(f'E{e:03d} L:{loss:.4f} V:{acc:.4f}')
    return best

# -------------------------
#  Single Experiment
# -------------------------
def run_experiment(test_geom='euclidean', seed=None, verbose=True, save_dir=None, run_num=None):
    """
    Run a single experiment comparing three GNN geometries.
    
    Args:
        test_geom: Geometry of test data
        seed: Random seed
        verbose: Print progress
        save_dir: Directory to save visualizations (None to skip)
        run_num: Run number for visualization filenames (None for single runs without index)
    """
    if seed is None:
        raise ValueError("run_experiment now requires a non-None seed.")
    if verbose:
        print(f"[INFO] Using fixed seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    x_test, A_test, y_test, tr_test, vl_test, te_test = get_graph_data(n=2000, nc=10, nf=128, geom=test_geom)
    res = {}
    
    for geom in ['euclidean','spherical','hyperbolic']:
        if verbose:
            print(f"\n--- {geom.upper()} GNN ---")
        if geom == test_geom:
            x, A, y, tr, vl, te = (
                x_test.clone(), A_test.clone(), y_test.clone(),
                tr_test.clone(), vl_test.clone(), te_test.clone()
            )
        else:
            x, A, y, tr, vl, te = get_graph_data(n=2000, nc=10, nf=128, geom=geom)

        model = GNN(x.shape[1], 256, y.max().item()+1, geom)
        t0 = time.time()
        train(model, x, A, y, (tr, vl), epochs=500, verbose=verbose)
        dur = time.time() - t0

        model.eval()
        with torch.no_grad():
            out = model(x_test, A_test)
            test_acc = (out.argmax(1)[te_test] == y_test[te_test]).float().mean().item()
            
            # Generate visualization if requested
            if save_dir is not None:
                # Get final embeddings from the model
                x_embed = model.c1(x_test, A_test)
                if run_num is not None:
                    viz_filename = f"{geom}_GNN-{test_geom}_test-{run_num}.png"
                else:
                    viz_filename = f"{geom}_GNN-{test_geom}_test.png"
                viz_path = os.path.join(save_dir, viz_filename)
                visualize_graph(x_embed, y_test, A_test, geom, viz_path, verbose=verbose)
        
        res[geom] = {'acc': test_acc, 'time': dur}
        if verbose:
            print(f"Test Accuracy on {test_geom} data: {test_acc:.4f}  (Time: {dur:.1f}s)")
    
    return res

# -------------------------
#  Summary Plotting
# -------------------------
def plot_summary(summary, save_dir, verbose=True):
    """
    Generate and save summary plots for multi-run results.
    
    Args:
        summary: Dictionary with statistics for each geometry
        save_dir: Directory to save the plot
        verbose: Print save confirmation
    """
    geoms = list(summary.keys())
    accs = [summary[g]['acc_mean'] for g in geoms]
    acc_err = [summary[g]['acc_std'] for g in geoms]
    times = [summary[g]['time_mean'] for g in geoms]
    time_err = [summary[g]['time_std'] for g in geoms]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(geoms, accs, yerr=acc_err, capsize=5)
    ax[0].set_title('Accuracy (mean ± std)')
    ax[0].set_ylabel('Accuracy')
    ax[1].bar(geoms, times, yerr=time_err, capsize=5, color='orange')
    ax[1].set_title('Runtime (s) (mean ± std)')
    ax[1].set_ylabel('Seconds')
    plt.tight_layout()
    
    summary_plot_path = os.path.join(save_dir, 'multi_run_results.png')
    plt.savefig(summary_plot_path, dpi=150)
    if verbose:
        print(f"[INFO] Saved plot: {summary_plot_path}")
    plt.close()

# -------------------------
#  Multi-run + Statistics
# -------------------------
def run_multiple(test_geom='euclidean', runs=3, verbose=True, save_outputs=True):
    """
    Run multiple experiments and aggregate results.
    
    Args:
        test_geom: Geometry of test data
        runs: Number of experimental runs
        verbose: Print progress
        save_outputs: Save plots, JSON, and visualizations
    """
    # Create timestamped directory for outputs
    save_dir = None
    if save_outputs:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = f"results_{test_geom}_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        if verbose:
            print(f"[INFO] Created output directory: {save_dir}")
    
    all_res = {g: {'acc': [], 'time': []} for g in ['euclidean','spherical','hyperbolic']}
    
    for i in range(runs):
        if verbose:
            print(f"\n===== RUN {i+1}/{runs} =====")
        seed = int(time.time()) % 10000 + i
        
        # Pass save_dir and run number for visualization
        run_num = i + 1
        res = run_experiment(
            test_geom=test_geom, 
            seed=seed, 
            verbose=verbose,
            save_dir=save_dir,
            run_num=run_num if save_outputs else None
        )
        
        for g in res:
            all_res[g]['acc'].append(res[g]['acc'])
            all_res[g]['time'].append(res[g]['time'])

    summary = {}
    for g in all_res:
        acc_mean = np.mean(all_res[g]['acc'])
        acc_std = np.std(all_res[g]['acc'])
        t_mean = np.mean(all_res[g]['time'])
        t_std = np.std(all_res[g]['time'])
        summary[g] = {'acc_mean': acc_mean, 'acc_std': acc_std, 'time_mean': t_mean, 'time_std': t_std}

    if verbose:
        print("\n=== SUMMARY ===")
        for g, v in summary.items():
            print(f"{g.capitalize():12s}: Acc={v['acc_mean']:.4f}±{v['acc_std']:.4f} | Time={v['time_mean']:.1f}±{v['time_std']:.1f}s")

    if save_outputs and save_dir:
        # Generate summary plot
        plot_summary(summary, save_dir, verbose=verbose)
        
        # Save JSON summary
        summary_json_path = os.path.join(save_dir, 'results_summary.json')
        with open(summary_json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        if verbose:
            print(f"[INFO] Saved summary: {summary_json_path}")
            print(f"[INFO] All outputs saved in: {save_dir}")

    return summary

# -------------------------
#  CLI Entry
# -------------------------
def main():
    parser = argparse.ArgumentParser(description='Multi-run GNN geometry comparison.')
    parser.add_argument('--test', choices=['euclidean','spherical','hyperbolic'], default='euclidean', help='Geometry of test data')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs to average over')
    parser.add_argument('--seed', type=int, default=None, help='Base random seed (optional)')
    parser.add_argument('--no-save', action='store_true', help='Disable saving plots and JSON outputs')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose printing')
    args = parser.parse_args()

    verbose = not args.quiet
    save_outputs = not args.no_save

    if args.seed is None:
        run_multiple(test_geom=args.test, runs=args.runs, verbose=verbose, save_outputs=save_outputs)
    else:
        if verbose:
            print(f"[INFO] Running single experiment with seed {args.seed}")
        
        # Create timestamped directory for single run if saving
        save_dir = None
        if save_outputs:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_dir = f"results_{args.test}_{timestamp}"
            os.makedirs(save_dir, exist_ok=True)
            if verbose:
                print(f"[INFO] Created output directory: {save_dir}")
        
        # Run single experiment
        results = run_experiment(
            test_geom=args.test, 
            seed=args.seed, 
            verbose=verbose,
            save_dir=save_dir,
            run_num=None  # No run number for single runs
        )
        
        # Save JSON with seed info if requested
        if save_outputs and save_dir:
            results_with_metadata = {
                'seed': args.seed,
                'test_geometry': args.test,
                'results': results
            }
            json_path = os.path.join(save_dir, 'single_run_results.json')
            with open(json_path, 'w') as f:
                json.dump(results_with_metadata, f, indent=2)
            if verbose:
                print(f"[INFO] Saved results: {json_path}")
                print(f"[INFO] All outputs saved in: {save_dir}")

if __name__ == '__main__':
    main()
