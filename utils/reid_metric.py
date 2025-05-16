# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        # self.plot_tsne(feats[:100], labels=self.pids)
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP

class NFormer_R1_mAP(Metric):
    def __init__(self, model, num_query, max_rank=50, feat_norm='yes'):
        super(NFormer_R1_mAP, self).__init__()
        self.model = model
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def plot_query_gallery_tsne(self, qf, gf, q_pids, g_pids, q_camids, g_camids, num_ids=10, random_state=42):
        """
        Plot t-SNE visualization showing both query and gallery images
        with their relationships for a random subset of person IDs.
        
        Parameters:
        -----------
        qf: Query features
        gf: Gallery features
        q_pids: Query person IDs
        g_pids: Gallery person IDs
        q_camids: Query camera IDs
        g_camids: Gallery camera IDs
        num_ids: Number of random person IDs to visualize (default: 10)
        random_state: Random seed for reproducibility
        """
        # Get all unique IDs from both query and gallery
        all_unique_pids = np.unique(np.concatenate([q_pids, g_pids]))
        
        # If we have fewer than num_ids unique people, use all of them
        if len(all_unique_pids) <= num_ids:
            selected_pids = all_unique_pids
        else:
            # Randomly select num_ids person IDs
            np.random.seed(random_state)
            # selected_pids = np.random.choice(all_unique_pids, size=num_ids, replace=False)
            selected_pids = [267, 305, 520, 625, 719, 756, 1006, 1108, 1438, 1482]
        
        print(f"Plotting t-SNE for {len(selected_pids)} randomly selected person IDs: {selected_pids}")
        
        # Filter query samples for selected IDs
        q_mask = np.isin(q_pids, selected_pids)
        filtered_qf = qf[q_mask]
        filtered_q_pids = q_pids[q_mask]
        filtered_q_camids = q_camids[q_mask]
        
        # Filter gallery samples for selected IDs
        g_mask = np.isin(g_pids, selected_pids)
        filtered_gf = gf[g_mask]
        filtered_g_pids = g_pids[g_mask]
        filtered_g_camids = g_camids[g_mask]
        
        # Check if we have enough data to plot
        if len(filtered_qf) == 0 or len(filtered_gf) == 0:
            print("Warning: No matching query or gallery samples found for the selected IDs. Cannot create t-SNE plot.")
            return
        
        # Combine query and gallery features but keep track of their source
        all_features = torch.cat([filtered_qf, filtered_gf], dim=0).cpu().numpy()
        
        # Create marker array (0 for query, 1 for gallery)
        markers = np.zeros(len(all_features))
        markers[len(filtered_qf):] = 1
        
        # Combine IDs
        all_pids = np.concatenate([filtered_q_pids, filtered_g_pids])
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=random_state)
        embeddings = tsne.fit_transform(all_features)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Define marker styles and colors
        marker_styles = ['o', 'x']  # circle for query, x for gallery
        
        # Get unique person IDs and assign colors
        unique_pids = np.unique(all_pids)
        colors = plt.cm.get_cmap('tab10', len(unique_pids))
        
        # Plot each identity group
        for i, pid in enumerate(unique_pids):
            for m_type in [0, 1]:  # 0=query, 1=gallery
                mask = (all_pids == pid) & (markers == m_type)
                if np.any(mask):
                    plt.scatter(
                        embeddings[mask, 0], 
                        embeddings[mask, 1],
                        c=[colors(i)], 
                        marker=marker_styles[m_type],
                        label=f"ID {pid} ({'Query' if m_type==0 else 'Gallery'})" if m_type==0 else None,
                        alpha=0.7,
                        s=80 if m_type==0 else 40  # Make query points larger
                    )
        
        # Draw lines connecting same identities between query and gallery
        # for pid in unique_pids:
        #     q_indices = np.where((filtered_q_pids == pid))[0]
        #     g_indices = np.where((filtered_g_pids == pid))[0]
            
        #     if len(q_indices) > 0 and len(g_indices) > 0:
        #         for q_idx in q_indices:
        #             q_point = embeddings[q_idx]
        #             for g_idx in g_indices:
        #                 g_point = embeddings[len(filtered_qf) + g_idx]
        #                 plt.plot([q_point[0], g_point[0]], [q_point[1], g_point[1]], 
        #                         'k-', alpha=0.2)
        
        plt.title(f't-SNE: Random {num_ids} Person IDs (Query-Gallery)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('query_gallery_tsne_random_ids_original.png', dpi=300, bbox_inches='tight')
        plt.show()

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # print(len(np.unique(self.pids)))
        # print(feats.shape)
        self.model.eval()
        with torch.no_grad():
            feats = self.model(feats.unsqueeze(0), stage='nformer')[0]
        self.model.train()
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        # self.plot_query_gallery_tsne(qf, gf, q_pids, g_pids, q_camids, g_camids)
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def plot_tsne(
        X: np.ndarray,
        labels: np.ndarray = None,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        random_state: int = 42,
        figsize: tuple = (8, 6),
        cmap: str = 'tab10'
    ) -> np.ndarray:
        """
        Compute a 2D t-SNE embedding of X and plot it.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
            Input feature matrix.
        labels : np.ndarray of shape (N,), optional
            Discrete labels for coloring points.
        perplexity : float
            t-SNE perplexity parameter.
        learning_rate : float
            t-SNE learning rate.
        n_iter : int
            Number of optimization iterations.
        random_state : int
            Random seed for reproducibility.
        figsize : tuple
            Figure size for the plot.
        cmap : str
            Colormap name for different labels.

        Returns
        -------
        X_embedded : np.ndarray, shape (N, 2)
            The 2D t-SNE embedding.
        """
        # 1) Fit t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=random_state,
            init='random'
        )
        X_embedded = tsne.fit_transform(X)

        # 2) Plot
        plt.figure(figsize=figsize)
        if labels is not None:
            unique_labels = np.unique(labels)
            for lbl in unique_labels:
                mask = labels == lbl
                plt.scatter(
                    X_embedded[mask, 0],
                    X_embedded[mask, 1],
                    label=str(lbl),
                    alpha=0.7
                )
            plt.legend(title='Label')
        else:
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.7)

        plt.title('t-SNE Projection')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('tsne_plot.png', dpi=300, bbox_inches='tight')
        plt.show()

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP
