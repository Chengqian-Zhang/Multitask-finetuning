import glob,os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
import seaborn as sns
import pandas as pd

def ana_representation(save_key):
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.linewidth': 2,
        'lines.linewidth': 2,
        'lines.markersize': 8
    })
    global_markersize = 9
    global_alpha = 0.6

    embed_info = np.load(f"../data/node_emb/{save_key}.npz", allow_pickle=True)
    rep = embed_info["rep"]
    atype = embed_info["atype"]
    k_nearest_elements = embed_info["k_nearest_elements"]
    k_nearest_distances = embed_info["k_nearest_distances"]
    env_category = embed_info["env_category"]
    rep_tsne = embed_info["rep_tsne"]

    cluster_score = davies_bouldin_score(rep_tsne, env_category)
 
    C_four_neighbor_mask = (env_category==0) # 2000
    C_three_neighbor_mask = (env_category==1) # 2000
    C_two_neighbor_mask = (env_category==2) # 2000
    H_mask = (env_category==3) # 2000
    F_mask = (env_category==4) # 591
    O_two_neighbor_mask = (env_category==5) # 2000
    O_one_neighbor_mask = (env_category==6) # 2000
    N_three_neighbor_mask = (env_category==7) # 2000
    N_two_neighbor_mask = (env_category==8) # 2000
    N_one_neighbor_mask = (env_category==9) # 1780

    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    ax.set_facecolor('white')

    c_colors = {
        'single': '#F5F5F5',
        'double': '#A9A9A9',
        'triple': '#696969'
    }
    h_color = '#FFE4B5'
    n_colors = {
        'single': '#ADD8E6',
        'double': '#4682B4',
        'triple': '#4169E1'
    }
    o_colors = {
        'single': '#FFA07A',
        'double': '#DC143C'
    }
    f_color = '#00CED1'

    categories = [
        (C_four_neighbor_mask, 'C_four_neighbor', c_colors['single']),
        (C_three_neighbor_mask, 'C_three_neighbor', c_colors['double']),
        (C_two_neighbor_mask, 'C_two_neighbor', c_colors['triple']),
        (H_mask, 'H', h_color),
        (F_mask, 'F', f_color),
        (O_two_neighbor_mask, 'O_two_neighbor', o_colors['single']),
        (O_one_neighbor_mask, 'O_one_neighbor', o_colors['double']),
        (N_three_neighbor_mask, 'N_three_neighbor', n_colors['single']),
        (N_two_neighbor_mask, 'N_two_neighbor', n_colors['double']),
        (N_one_neighbor_mask, 'N_one_neighbor', n_colors['triple'])
    ]

    data_list = []
    for mask, label, color in categories:
        indices = np.where(mask)[0]
        for idx in indices:
            data_list.append({
                't-SNE1': rep_tsne[idx, 0],
                't-SNE2': rep_tsne[idx, 1],
                'Category': label,
                'Element': label.split('_')[0]
            })

    df = pd.DataFrame(data_list)

    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.2)

    color_dict = {label: color for _, label, color in categories}

    for mask, label, color in categories:
        subset = df[df['Category'] == label]
        if len(subset) > 0:
            ax.scatter(
                subset['t-SNE1'], 
                subset['t-SNE2'],
                s=global_markersize**2 * 1.2,
                alpha=global_alpha,
                color=color,
                edgecolors='black',
                linewidths=0.3,
                label=label
            )

    for spine in ax.spines.values():
        spine.set_linewidth(4)

    ax.grid(True, alpha=0.2, linestyle='--')
    ax.tick_params(axis='x', length=0, width=3, direction="in", pad=15)
    ax.tick_params(axis='y', length=0, width=3, direction="in", pad=15)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.15])

    plt.tight_layout()

    if save_key == "pretrain":
        plt.savefig(f"figure2d.png", 
                dpi=300,
                bbox_inches='tight', 
                facecolor='white',
                edgecolor='none')
    elif save_key == "ft":
        plt.savefig(f"figure2e.png", 
                dpi=300,
                bbox_inches='tight', 
                facecolor='white',
                edgecolor='none')
    elif save_key == "mft":
        plt.savefig(f"figure2f.png", 
                dpi=300,
                bbox_inches='tight', 
                facecolor='white',
                edgecolor='none')
    else:
        raise RuntimeError("Save key is wrong!")

    return cluster_score

if __name__ == "__main__":
    db_index_dict = {}
    for save_key in ["pretrain", "ft", "mft"]:
        db_index = ana_representation(save_key)
        db_index_dict[save_key] = db_index
    print("DB index:", )
    print("Pretrain: 1.0")
    print("FT: ", round(db_index_dict["ft"]/db_index_dict["pretrain"],1))
    print("LP: ", round(db_index_dict["mft"]/db_index_dict["pretrain"],1))