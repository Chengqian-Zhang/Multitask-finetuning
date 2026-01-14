import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob,os,json

task_dict = {
   "HOMO": {
      "id_scratch": 0.002953668115074033,
      "ood_scratch": 0.015478466456947665,
      "id_scratch_list": [
         0.0029860657207458,
         0.002908251502977,
         0.0029666871214993
      ],
      "ood_scratch_list": [
         0.0165783476721863,
         0.014869361802605,
         0.0149876898960517
      ],
      "id_ft": 0.0022016464125341334,
      "ood_ft": 0.015645072526071866,
      "id_ft_list": [
         0.0022892604132623,
         0.0021406338891744,
         0.0021750449351657
      ],
      "ood_ft_list": [
         0.0158246336734677,
         0.015812649811902,
         0.0152979340928459
      ],
      "id_lp": 0.004552509072060734,
      "ood_lp": 0.012459338348675101,
      "id_lp_list": [
         0.0045620444649554,
         0.0045859903763508,
         0.004509492374876
      ],
      "ood_lp_list": [
         0.0126249901138246,
         0.0121325684623539,
         0.0126204564698468
      ],
      "id_mft": 0.0021271476576196664,
      "id_mft_list": [
         0.002114269305809,
         0.0021329652723919,
         0.0021342083946581
      ],
      "ood_mft": 0.0133911323139597,
      "ood_mft_list": [
         0.0136159447711035,
         0.0130014429136746,
         0.013556009257101
      ]
   },
   "LUMO": {
      "id_scratch": 0.0028875657765656,
      "ood_scratch": 0.012990744545538868,
      "id_scratch_list": [
         0.0029622275660066,
         0.0028986510189587,
         0.0028018187447315
      ],
      "ood_scratch_list": [
         0.0128680920433471,
         0.0125505048341128,
         0.0135536367591567
      ],
      "id_ft": 0.0022554110240224335,
      "ood_ft": 0.013487655101857933,
      "id_ft_list": [
         0.0022807072066211,
         0.0022546018856128,
         0.0022309239798334
      ],
      "ood_ft_list": [
         0.0141544086906608,
         0.0134669509506555,
         0.0128416056642575
      ],
      "id_lp": 0.0053871872455642,
      "ood_lp": 0.0127772189736659,
      "id_lp_list": [
         0.0052886076558324,
         0.0054868512181362,
         0.005386102862724
      ],
      "ood_lp_list": [
         0.0131177455291621,
         0.0122221517929129,
         0.0129917595989227
      ],
      "id_mft": 0.0023599873351582332,
      "id_mft_list": [
         0.0023550667175236,
         0.002360929013141,
         0.0023639662748101
      ],
      "ood_mft": 0.010695506285296632,
      "ood_mft_list": [
         0.0111251399550543,
         0.0102636039178765,
         0.0106977749829591
      ]
   },
   "GAP": {
      "id_scratch": 0.004546591272718434,
      "ood_scratch": 0.024350155754808565,
      "id_scratch_list": [
         0.0046015713107313,
         0.0045941876238705,
         0.0044440148835535
      ],
      "ood_scratch_list": [
         0.0237429352517156,
         0.0247842489431209,
         0.0245232830695892
      ],
      "id_ft": 0.0033842678054535,
      "ood_ft": 0.020100331103820566,
      "id_ft_list": [
         0.003409137231318,
         0.0033718766668283,
         0.0033717895182142
      ],
      "ood_ft_list": [
         0.0197967420231043,
         0.0205383664434579,
         0.0199658848448995
      ],
      "id_lp": 0.007729204832090267,
      "ood_lp": 0.0192880937937861,
      "id_lp_list": [
         0.0076490127548774,
         0.0077830894950781,
         0.0077555122463153
      ],
      "ood_lp_list": [
         0.0194641297671017,
         0.0192759661395934,
         0.0191241854746632
      ],
      "id_mft": 0.0033749145107645,
      "id_mft_list": [
         0.0033648134192354,
         0.003356892218048,
         0.0034030378950101
      ],
      "ood_mft": 0.0171329881722055,
      "ood_mft_list": [
         0.0165352947218215,
         0.0173643687643149,
         0.0174993010304801
      ]
   },
   "ZPVE": {
      "id_scratch": 0.00013352375694826666,
      "ood_scratch": 0.00028511373592446664,
      "id_scratch_list": [
         0.0001358337505039,
         0.0001337449012512,
         0.0001309926190897
      ],
      "ood_scratch_list": [
         0.0002884532291382,
         0.0002813492809494,
         0.0002855386976858
      ],
      "id_ft": 0.00010217033586216666,
      "ood_ft": 0.00026215006260086666,
      "id_ft_list": [
         0.0001026644780971,
         0.0001014125545793,
         0.0001024339749101
      ],
      "ood_ft_list": [
         0.0002683916189525,
         0.0002638163481741,
         0.000254242220676
      ],
      "id_lp": 0.00016279505053703333,
      "ood_lp": 0.00029261082168920003,
      "id_lp_list": [
         0.0001630995005266,
         0.0001619140523944,
         0.0001633715986901
      ],
      "ood_lp_list": [
         0.0002863923665247,
         0.0002948276527067,
         0.0002966124458362
      ],
      "id_mft": 0.00010323008436783334,
      "id_mft_list": [
         0.0001042318355999,
         0.0001032732002814,
         0.0001021852172222
      ],
      "ood_mft": 0.0002536941571028667,
      "ood_mft_list": [
         0.0002480350433281,
         0.0002472319347555,
         0.000265815493225
      ]
   },
   "<$R^2$>": {
      "id_scratch": 8.50168568401577,
      "ood_scratch": 21.418059487379338,
      "id_scratch_list": [
         7.966710511689315,
         9.300527364744266,
         8.237819175613735
      ],
      "ood_scratch_list": [
         20.20794301773781,
         20.68303989616949,
         23.36319554823072
      ],
      "id_ft": 8.105129657188682,
      "ood_ft": 18.653324548436807,
      "id_ft_list": [
         8.133162472247479,
         8.100578252598076,
         8.081648246720494
      ],
      "ood_ft_list": [
         20.33947691161936,
         22.051307041763785,
         13.569189691927273
      ],
      "id_lp": 8.812388858842297,
      "ood_lp": 14.187015407427046,
      "id_lp_list": [
         8.804562772484891,
         8.812044025439338,
         8.820559778602664
      ],
      "ood_lp_list": [
         14.19278494758612,
         14.202740039889072,
         14.165521234805944
      ],
      "id_mft": 8.291216225872509,
      "id_mft_list": [
         8.257362329066437,
         8.323329653349768,
         8.292956695201323
      ],
      "ood_mft": 13.689393000016556,
      "ood_mft_list": [
         13.801527507498497,
         13.754650501054224,
         13.512000991496947
      ]
   },
   "$\\alpha$": {
      "id_scratch": 0.3646871722143998,
      "ood_scratch": 1.907266532463303,
      "id_scratch_list": [
         0.359100848566995,
         0.3785763123418169,
         0.3563843557343873
      ],
      "ood_scratch_list": [
         1.875902222021736,
         1.9522570515298083,
         1.893640323838365
      ],
      "id_ft": 0.29713191496461944,
      "ood_ft": 1.6386012795086515,
      "id_ft_list": [
         0.295314685518615,
         0.2969237942011731,
         0.2991572651740701
      ],
      "ood_ft_list": [
         1.6831854402914814,
         1.5888828069301202,
         1.643735591304353
      ],
      "id_lp": 0.4583917828791319,
      "ood_lp": 1.640966391664124,
      "id_lp_list": [
         0.4587420144528292,
         0.4584470292695524,
         0.4579863049150141
      ],
      "ood_lp_list": [
         1.6398862504999967,
         1.6622051820871548,
         1.6208077424052203
      ],
      "id_mft": 0.29573661442352067,
      "id_mft_list": [
         0.2938416609794308,
         0.2973049308164944,
         0.2960632514746367
      ],
      "ood_mft": 1.5355949278530086,
      "ood_mft_list": [
         1.53725639110085,
         1.5453735905556296,
         1.5241548019025466
      ]
   },
   "$\\mu$": {
      "id_scratch": 0.2937220871439103,
      "ood_scratch": 2.0867455739053455,
      "id_scratch_list": [
         0.3001380096564583,
         0.2929128162276274,
         0.2881154355476452
      ],
      "ood_scratch_list": [
         2.079409210464791,
         2.0902834605267406,
         2.090544050724505
      ],
      "id_ft": 0.27559318703527946,
      "ood_ft": 2.267442795071108,
      "id_ft_list": [
         0.2766091611076152,
         0.2729748384564738,
         0.2771955615417494
      ],
      "ood_ft_list": [
         2.2311220067145285,
         2.264160392667336,
         2.3070459858314605
      ],
      "id_lp": 0.30891522005001454,
      "ood_lp": 1.7469540550673797,
      "id_lp_list": [
         0.3086318029941169,
         0.3085213772562724,
         0.3095924798996544
      ],
      "ood_lp_list": [
         1.742700710253421,
         1.7262532150664125,
         1.7719082398823052
      ],
      "id_mft": 0.2778854458672824,
      "id_mft_list": [
         0.2773150478995076,
         0.2757899260882777,
         0.2805513636140619
      ],
      "ood_mft": 2.0867070843667364,
      "ood_mft_list": [
         2.1015964507804616,
         2.0688194076185527,
         2.0897053947011943
      ]
   },
   "$C_v$": {
      "id_scratch": 0.11613710614533296,
      "ood_scratch": 0.19984769242321043,
      "id_scratch_list": [
         0.1175327454822405,
         0.1156781488139482,
         0.1152004241398102
      ],
      "ood_scratch_list": [
         0.2073781263561299,
         0.2068149865739777,
         0.1853499643395236
      ],
      "id_ft": 0.09671518251780624,
      "ood_ft": 0.18838946513273216,
      "id_ft_list": [
         0.0983728972222557,
         0.0958802230851575,
         0.0958924272460055
      ],
      "ood_ft_list": [
         0.1901195873099793,
         0.1907585430023346,
         0.1842902650858827
      ],
      "id_lp": 0.1494972454817618,
      "ood_lp": 0.2808165872435225,
      "id_lp_list": [
         0.1492765051979043,
         0.1494292111836611,
         0.14978602006372
      ],
      "ood_lp_list": [
         0.2841193005246238,
         0.2760812881945598,
         0.2822491730113838
      ],
      "id_mft": 0.09655949467669674,
      "id_mft_list": [
         0.0972983338944208,
         0.0972157165689553,
         0.0951644335667141
      ],
      "ood_mft": 0.18147236346765708,
      "ood_mft_list": [
         0.1799113120251762,
         0.1819669127079004,
         0.1825388656698947
      ]
   }
}

global_fontsize = 35
global_linewidth = 4
plt.rcParams['font.size'] = global_fontsize

id_color = "#87e885"
ood_color = "#3cb9fc"

tasks = [str(ii) for ii in list(task_dict.keys())]
id_scratch_result = np.array([task_dict[ii]["id_scratch"] for ii in tasks])
id_ft_result = np.array([task_dict[ii]["id_ft"] for ii in tasks])
id_lp_result = np.array([task_dict[ii]["id_lp"] for ii in tasks])
id_mft_result = np.array([task_dict[ii]["id_mft"] for ii in tasks])
ood_scratch_result = np.array([task_dict[ii]["ood_scratch"] for ii in tasks])
ood_ft_result = np.array([task_dict[ii]["ood_ft"] for ii in tasks])
ood_lp_result = np.array([task_dict[ii]["ood_lp"] for ii in tasks])
ood_mft_result = np.array([task_dict[ii]["ood_mft"] for ii in tasks])

id_ft_compare_scratch = 100*(id_scratch_result - id_ft_result)/id_scratch_result
ood_ft_compare_scratch = 100*(ood_scratch_result - ood_ft_result)/ood_scratch_result

id_lp_compare_scratch = 100*(id_scratch_result - id_lp_result)/id_scratch_result
ood_lp_compare_scratch = 100*(ood_scratch_result - ood_lp_result)/ood_scratch_result

id_mft_compare_scratch = 100*(id_scratch_result - id_mft_result)/id_scratch_result
ood_mft_compare_scratch = 100*(ood_scratch_result - ood_mft_result)/ood_scratch_result

id_mft_compare_sft = 100*(id_ft_result - id_mft_result)/id_ft_result
ood_mft_compare_sft = 100*(ood_ft_result - ood_mft_result)/ood_ft_result

x = np.arange(len(tasks))
x_2 = np.arange(len(tasks)) + len(tasks)
x_3 = np.arange(len(tasks)) + 2 * len(tasks)
x_all = np.arange(len(tasks) * 3)

width = 0.3

fig, ax = plt.subplots(figsize=(30, 10), dpi=300)

bars1 = ax.bar(x - width, id_ft_compare_scratch , width, 
               label='ID', 
               color=id_color, edgecolor='black')
bars2 = ax.bar(x, ood_ft_compare_scratch , width, 
               label='OOD', 
               color=ood_color, edgecolor='black')
bars3 = ax.bar(x_2 - width, id_lp_compare_scratch, width,  
               color=id_color, edgecolor='black')
bars4 = ax.bar(x_2, ood_lp_compare_scratch, width, 
               color=ood_color, edgecolor='black')
bars5 = ax.bar(x_3 - width, id_mft_compare_scratch, width,  
               color=id_color, edgecolor='black')
bars6 = ax.bar(x_3, ood_mft_compare_scratch, width,
               color=ood_color, edgecolor='black')

ax.set_ylabel('Relative Improvement (%)', fontsize=global_fontsize)
ax.set_xticks(x_all)
ax.set_ylim(-19.9, 60)
ax.set_xlim(-width*2, 3 * len(tasks)-width * 2)

ax.grid(True, axis='y', alpha=1, linestyle='--')
ax.axhline(y=0, color='black', linewidth=global_linewidth)
ax.axvline(x=len(tasks)-width*2.2, color='black', linewidth=global_linewidth)
ax.axvline(x=len(tasks)*2-width*2.2, color='black', linewidth=global_linewidth)

fontweight='bold'
ax.text(len(tasks)/2-0.5, ax.get_ylim()[1]*0.95, f'(a)FT vs Scratch\nID: {np.mean(id_ft_compare_scratch):.1f}%\nOOD: {np.mean(ood_ft_compare_scratch):.1f}%', 
        ha='center', va='top', fontsize=global_fontsize, fontweight='bold')
ax.text(len(tasks)*1.5-0.5, ax.get_ylim()[1]*0.95, f'(b)LP vs Scratch\nID: {np.mean(id_lp_compare_scratch):.1f}%\nOOD: {np.mean(ood_lp_compare_scratch):.1f}%', 
        ha='center', va='top', fontsize=global_fontsize,fontweight='bold')
ax.text(len(tasks)*2.5-0.5, ax.get_ylim()[1]*0.95, f'(c)MFT vs Scratch\nID: {np.mean(id_mft_compare_scratch):.1f}%\nOOD: {np.mean(ood_mft_compare_scratch):.1f}%', 
        ha='center', va='top', fontsize=global_fontsize,fontweight='bold')

for spine in ax.spines.values():
    spine.set_linewidth(global_linewidth)

all_labels = tasks + tasks + tasks
ax.set_xticklabels(all_labels, fontsize=global_fontsize, rotation=60)
ax.tick_params(axis='x', which='both', length=10, width=3, direction="in", pad=10)
ax.tick_params(axis='y', length=0)

plt.tight_layout(rect=[0, 0, 1, 1])

plt.savefig("figure2abc.png", dpi=300, bbox_inches='tight')