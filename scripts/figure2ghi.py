import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
import seaborn as sns
import pandas as pd

def ana_descriptor(save_key):
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.linewidth': 2,
        'lines.linewidth': 2,
        'lines.markersize': 8
    })
    global_markersize = 9
    global_alpha = 0.6

    embed_info = np.load(f"../data/edge_emb/{save_key}.npz", allow_pickle=True)
    g2s = embed_info["g2s"]
    pairs = embed_info["pairs"]
    pair_distance = embed_info["pair_distance"]
    rep_tsne = embed_info["rep_tsne"]

    cluster_score = davies_bouldin_score(rep_tsne, [0] * 2000 + [1] * 2000 + [2] * 2000 + [3] * 2000+ [4] * 2000 + [5] * 2000)
    
    bond_radii = {
            'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66,
            'F': 0.57, 'P': 1.07, 'S': 1.05, 'Cl': 1.02
    }
    bond_factor = 1.15

    CO_mask_near = (np.all(pairs == [5, 7], axis=1) | np.all(pairs == [7, 5], axis=1)) & (pair_distance < bond_factor * (bond_radii["C"] + bond_radii["O"])) # 2000
    CO_mask_middle = (np.all(pairs == [5, 7], axis=1) | np.all(pairs == [7, 5], axis=1)) & (pair_distance >3) & (pair_distance < 4) # 2000
    CO_mask_far = (np.all(pairs == [5, 7], axis=1) | np.all(pairs == [7, 5], axis=1)) & (pair_distance > 5) & (pair_distance < 6) # 2000

    CH_mask_near = (np.all(pairs == [5, 0], axis=1) | np.all(pairs == [0, 5], axis=1)) & (pair_distance < bond_factor * (bond_radii["C"] + bond_radii["H"])) # 2000
    CH_mask_middle = (np.all(pairs == [5, 0], axis=1) | np.all(pairs == [0, 5], axis=1)) & (pair_distance >3) & (pair_distance < 4) # 2000
    CH_mask_far = (np.all(pairs == [5, 0], axis=1) | np.all(pairs == [0, 5], axis=1)) & (pair_distance >5)  & (pair_distance < 6) # 2000

    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    ax.set_facecolor('white')

    ch_colors = {
        'near': '#00FFFF',
        'middle': '#40A0A0',
        'far': '#0080FF'
    }

    co_colors = {
        'near': '#FFDAB9',
        'middle': '#FF8000',
        'far': '#FF0000'
    }

    categories = [
        (CO_mask_near, 'CO_near', co_colors['near']),
        (CO_mask_middle, 'CO_middle', co_colors['middle']),
        (CO_mask_far, 'CO_far', co_colors['far']),
        (CH_mask_near, 'CH_near', ch_colors["near"]),
        (CH_mask_middle, 'CH_middle', ch_colors["middle"]),
        (CH_mask_far, 'CH_far', ch_colors['far'])
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
        plt.savefig(f"figure2g.png", 
                dpi=300, 
                bbox_inches='tight', 
                facecolor='white', 
                edgecolor='none') 
    elif save_key == "ft":
        plt.savefig(f"figure2h.png",
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none') 
    elif save_key == "mft":
        plt.savefig(f"figure2i.png",
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
        db_index = ana_descriptor(save_key)
        db_index_dict[save_key] = db_index
    print("DB index:", )
    print("Pretrain: 1.0")
    print("FT: ", round(db_index_dict["ft"]/db_index_dict["pretrain"], 1))
    print("LP: ", round(db_index_dict["mft"]/db_index_dict["pretrain"], 1))