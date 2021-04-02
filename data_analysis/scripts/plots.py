import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt


def concat_datasets_for_ploting(batch_patch, upper_patch, bracis_patch, minasbr_patch):
    df_batch = pd.read_csv(batch_patch)
    df_upper = pd.read_csv(upper_patch)

    df_bracis = pd.read_csv(bracis_patch)
    df_bracis_classifier = df_bracis.query("Classifier == 'MINAS-BR'").copy()
    df_bracis_unkrm = df_bracis.query("Classifier == 'UnkRM'").copy()
    df_bracis_classifier["Classifier"] = "MINAS-BR_v1"
    df_bracis_unkrm["Classifier"] = "UnkRM_v1"
    # df_bracis = pd.concat([df_bracis_classifier, df_bracis_unkrm])
    df_bracis = pd.concat([df_bracis_classifier.iloc[:-1], df_bracis_unkrm.iloc[:-1]])

    df_minasbr = pd.read_csv(minasbr_patch)
    df_minasbr_classifier = df_minasbr.query("Classifier == 'MINAS-BR'")
    df_minasbr_unkrm = df_minasbr.query("Classifier == 'UnkRM'")
    df_minasbr_classifier["Classifier"] = "MINAS-BR_v2"
    df_minasbr_unkrm["Classifier"] = "UnkRM_v2"
    # df_minasbr = pd.concat([df_minasbr_classifier,df_minasbr_unkrm])
    df_minasbr = pd.concat([df_minasbr_classifier.iloc[:-1], df_minasbr_unkrm.iloc[:-1]])

    df_final = pd.concat([df_minasbr, df_bracis, df_upper, df_batch])
    return df_final



def get_concept_evolution_np_info(minasbr_path):
    concept_evolution_info = pd.read_csv(os.path.join(minasbr_path,"conceptEvolution-info.csv"))

    np_windows = pd.read_csv(os.path.join(minasbr_path,"NP-info.csv"))
    print(np_windows)
    if len(np_windows) > 0:
        np_windows = np_windows.dropna().reset_index().drop(['index'],axis=1)
        np_windows.columns = ["timestamp","NP","associated_class","timestamp_association","window_association"]
        x = np_windows['window_association'].duplicated().map(lambda i: not i)
        np_windows_duplicated_less = np_windows[x]
        np_windows_duplicated_less.reset_index(inplace=True)
        np_windows_duplicated_less = np_windows_duplicated_less.drop(['index'],axis=1)
        np_info = np_windows_duplicated_less.sort_values(by='window_association')
        return np_info, concept_evolution_info
    else:
        return None, concept_evolution_info

def plot_measures_over_time_sinais(data,
                                   np_info,
                                   np_info_v1,
                                   concept_evolution_info,
                                   linewidth,
                                   markersize,
                                   savefig=False,
                                   name_fig=None):
    sns.set_style("ticks")

    fig, axs = plt.subplots(3, gridspec_kw={
        'width_ratios': [1],
        'height_ratios': [0.75, 0.75, 7],
        'hspace': 0.15})

    sns.set(font_scale=1)
    sns.lineplot(data=data,
                 y="F1M",
                 x="Timestamp",
                 hue="Classifier",
                 style="Classifier",
                 markers=True,
                 dashes=True,
                 linewidth=linewidth,
                 markersize=markersize,
                 ax=axs[2])

    # fig.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    axs[2].legend(fontsize='x-large',
                  bbox_to_anchor=(0., -0.1, 1., -0.1),
                  loc='best',
                  ncol=3,
                  mode="expand",
                  labelspacing=1,
                  borderaxespad=0.)

    axs[2].set_ylim(0, 1)
    axs[2].grid(color='lightgray', linestyle='-', linewidth=0.5)
    axs[2].set_xlim(0, 51)
    axs[0].set_xlim(0, 51)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_xlim(0, 51)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    fig.set_size_inches(12, 6)
    x = 0
    pos = [-30, -5, 20, 45, 60, 85, 70, 95, 120, 120, 110, 140, 100, 100, 100, 100, 100]
    for i in concept_evolution_info.index:
        axs[0].annotate(concept_evolution_info['label'].iloc[i],
                        # xy=(concept_evolution_info['window'].iloc[i], 0),
                        xy=(concept_evolution_info['window'].iloc[i]+1, 0),
                        xycoords='data',
                        bbox=dict(boxstyle="round", fc="red", ec="red"),
                        xytext=(pos[x], 30),
                        textcoords='offset points',
                        ha='center',
                        arrowprops=dict(arrowstyle="->",color="gray"),
                        fontsize=12)
        x+=1

        axs[0].vlines(x=concept_evolution_info['window']+1,
        # axs[0].vlines(x=concept_evolution_info['window'],
                      ymin=0,
                      ymax=1,
                      colors='black',
                      linestyles='solid')

    if np_info is not None:
        for i in np_info.index:
            # label = str(np_windows_duplicated_less['associated_class'].iloc[i])
            axs[1].annotate(int(np_info['associated_class'].iloc[i]),
                            xy=(np_info.iloc[i, -1], 1),
                            xycoords='data',
                            xytext=(30, 10),
                            textcoords='offset points',
                            ha='center',
                            arrowprops=dict(arrowstyle="simple"),
                            bbox=dict(boxstyle="round", fc="blue", ec="white"),
                            fontsize=12)

            axs[1].vlines(x=np_info.iloc[i, -1],
                          ymin=0,
                          ymax=1,
                          colors='black',
                          linestyles='solid',
                          label="NP1")

    if np_info_v1 is not None:
        x = 0
        for i in np_info_v1.index:

            axs[1].vlines(x=np_info_v1.iloc[i, -1],
                          ymin=0,
                          ymax=1,
                          colors='red',
                          linestyles='dashed',
                          label="NP1")

            axs[1].annotate(int(np_info_v1['associated_class'].iloc[i]),
                            xy=(np_info_v1.iloc[i, -1], 0),
                            xycoords='data',
                            xytext=(-15+x, -10),
                            textcoords='offset points',
                            ha='center',
                            arrowprops=dict(arrowstyle="simple"),
                            bbox=dict(boxstyle="round", fc="green", ec="white"),
                            fontsize=12)
            x += 30

    axs[2].set_xlabel("Windows", fontsize=18)
    axs[2].set_ylabel("F1M and UnkRM", fontsize=18)
    axs[2].tick_params(labelsize=12)

    if savefig:
        plt.savefig(name_fig, dpi=300, format='pdf', bbox_inches='tight')
    plt.show()