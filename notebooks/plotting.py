import seaborn as sns


def plot_f_post(means,
                cov,
                G,
                ax,
                color_palette,
                offset=0,
                aaa=None, title=''):
    if aaa is None:
        aaa = [0] * means.shape[0]
    for i, c in zip(range(means.shape[0]), color_palette):
        ax.plot(G[i], means[i] + offset * i + aaa[i], color=c)
        top = means[i][:, 0] + 2.0 * (cov[i][:, 0]) ** 0.5 + offset * i + aaa[i]
        bot = means[i][:, 0] - 2.0 * (cov[i][:, 0]) ** 0.5 + offset * i + aaa[i]
        ax.fill_between(G[i], top, bot, color=c, alpha=0.3)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(title)


def plot_data_fit(inputs,
                  outputs,
                  inputs_miss,
                  outputs_miss,
                  inputs_new,
                  means,
                  cov,
                  var,
                  ax,
                  color_palette,
                  title='',
                  offset=0.4,
                  to_plot=None,
                  aaa=None):

    if to_plot is None:
        to_plot = list(range(means.shape[0]))
    count = 0
    if aaa is None:
        aaa = [0] * len(to_plot)
    for i in to_plot:
        c = color_palette[i]
        top = means[i][:, 0] + 2.0 * (cov[i][:, 0] + var) ** 0.5
        bot = means[i][:, 0] - 2.0 * (cov[i][:, 0] + var) ** 0.5
        ax.plot(inputs_new[i][:, -1], means[i] + offset * count + aaa[count], color=c)
        ax.scatter(inputs[i][:, -1], outputs[i] + offset * count + aaa[count], color=c, marker='.', s=20)
        ax.fill_between(inputs_new[i][:, -1], top + offset * count + aaa[count], bot + offset * count + aaa[count],
                         color=c, alpha=0.3)
        ax.scatter(inputs_miss[i][:, -1], outputs_miss[i] + offset * count + aaa[count], color=c, marker='*')

        ax.scatter(inputs_miss[i][:, -1], outputs_miss[i] + offset * count + aaa[count], color='black', marker='.', s=15)
        count += 1

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(title)


def plot_latent(Z,
                latent_posterior,
                ax,
                color_palette,
                title):
    ax.set_aspect(1)

    z1, z2, pz = latent_posterior
    cm = sns.color_palette('bone', as_cmap=True)
    ax.pcolormesh(z1, z2, pz, shading='gouraud', cmap=cm) # plt.cm.Blues

    for i, c in zip(range(Z.shape[0]), color_palette):
        ax.scatter(Z[i, 0], Z[i, 1], color=c)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(title)
