"""Optional figures for prescribed-flow probability and particle diagnostics."""
import numpy as np


_LABELS = ('Mean w′', 'Variance of w′', 'Mean Δw′ / Δt', 'Variance of Δw′ / Δt')
_LES_LABELS = ('Subfilter variance of w′', 'Covariance of u′ and w′', _LABELS[-1])


def write_profile_figures(output, stem, cases, *, ghosts=False, les=False):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    def figure(width, height):
        result = Figure(figsize=(width, height), layout='constrained')
        FigureCanvasAgg(result)
        return result

    probabilities = figure(3*len(cases), 4.5)
    variants = [('', _LABELS)]
    if ghosts:
        variants.append(('_showGhost', _LABELS))
    if les:
        variants.append(('_LES', _LES_LABELS))
    plots = {}
    for suffix, labels in variants:
        canvas = figure(10, 5)
        plots[suffix] = canvas, canvas.subplots(1, len(labels), sharey=True), labels
    for i, (step, case) in enumerate(cases.items()):
        diagnostics = case['diagnostics']
        with np.load(diagnostics['arrays']) as arrays:
            ax = probabilities.add_subplot(1, len(cases), i+1)
            ax.stairs(arrays['relative_probability'], arrays['probability_edges'],
                      orientation='horizontal', fill=True, alpha=.65)
            ax.axvline(1, color='black', linestyle='--', linewidth=1)
            ax.set(xlabel='Relative probability', ylim=(0, 1),
                   title=f"Δt = {step} s · S = {diagnostics['entropy']:.3f}")
            if i == 0:
                ax.set_ylabel('z / L')
            for suffix, (_, axes, labels) in plots.items():
                prefix = 'les_' if suffix == '_LES' else ''
                z = arrays['reference_z']
                limits = (z[0], z[-1]) if suffix == '_showGhost' else (0, 1)
                for j, axis in enumerate(axes):
                    if i == 0:
                        axis.plot(arrays[prefix+'reference_profiles'][j], z, color='black', label='Expected')
                        axis.set(xlabel=labels[j], ylim=limits)
                    axis.plot(arrays[prefix+'observed'][j], arrays[prefix+'statistics_z'], '.-', markersize=3,
                              label=f'{step} s')
    path = output/f'{stem}_probabilityPlots.pdf'
    probabilities.savefig(path)
    paths = [str(path)]
    for suffix, (canvas, axes, _) in plots.items():
        axes[0].set_ylabel('z / L')
        axes[-1].legend()
        path = output/f'{stem}_statisticPlots{suffix}.pdf'
        canvas.savefig(path)
        paths.append(str(path))
    return paths
