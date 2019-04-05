
import matplotlib.pyplot as plt
plt.style.use('ggplot')


### This will function will create the actual estimations vs predicted values
def plot_many_predicteds_vs_actuals(var_names, y_hat, n_bins=50):
    fig, axs = plt.subplots(len(var_names), figsize=(12, 3*len(var_names)))
    for ax, name in zip(axs, var_names):
        x = df_new_train[name]
        predicteds_vs_actuals(ax, x, df_new_train["cycles_to_fail"], y_hat, n_bins=n_bins)
        # ax.set_title("{} Predicteds vs. Actuals".format(name))
    return fig, axs


