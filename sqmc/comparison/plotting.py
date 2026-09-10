"""Offline publication plots for fresh-scramble QMC and shared Hilbert sorting."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']


def axes_style(ax):
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator([128, 2048, 32768, 131072]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: rf'$2^{{{int(np.log2(x))}}}$'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('Points N')
    ax.grid(True, which='major', alpha=.15)


def qmc_figure(rows):
    if any(r.get('scramble') is not True or r['mode'] != 'fresh' for r in rows):
        raise ValueError('QMC publication figure requires fresh scrambling')
    dims = sorted({r['dimension'] for r in rows})
    styles = [('jax', 'gpu', '-', 'JAX GPU'), ('scipy', 'cpu', ':', 'SciPy CPU')]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), layout='constrained')
    for sequence, pair in zip(['sobol', 'halton'], axes):
        for d, color in zip(dims, COLORS):
            grouped = {}
            for impl, backend, style, label in styles:
                values = sorted((r for r in rows if r['sequence'] == sequence and r['dimension'] == d and r['implementation'] == impl and r['backend'] == backend), key=lambda r:r['n'])
                grouped[(impl, backend)] = {r['n']:r['median_seconds'] for r in values}
                pair[0].plot([r['n'] for r in values], [r['median_seconds']*1000 for r in values], color=color, linestyle=style, linewidth=1.1)
            gpu = grouped[('jax', 'gpu')]
            for identity, style in [(('scipy', 'cpu'), '--')]:
                values = grouped[identity]
                counts = sorted(set(values) & set(gpu))
                pair[1].plot(counts, [values[n]/gpu[n] for n in counts], color=color, linestyle=style, linewidth=1.1)
        pair[0].set(title=sequence.title() + ': fresh scrambling + sampling', ylabel='Median time (ms)', yscale='log')
        pair[1].set(title=sequence.title() + ': SciPy CPU / JAX GPU', ylabel='Median-runtime ratio', yscale='log')
        if not any(r['sequence'] == sequence and r['backend'] == 'gpu' for r in rows):
            pair[1].set(xlim=(min(r['n'] for r in rows), max(r['n'] for r in rows)*1.01), ylim=(.5, 2))
            pair[1].text(.5, .5, 'GPU not requested', transform=pair[1].transAxes, ha='center')
        pair[1].axhline(1, color='#555555', linestyle='-.', linewidth=.8)
        for ax in pair:
            axes_style(ax)
    handles = [Line2D([0],[0],color=c,label=f'd = {d}') for d,c in zip(dims,COLORS)]
    handles += [Line2D([0],[0],color='black',linestyle=style,label=label) for _,_,style,label in styles]
    fig.legend(handles=handles,loc='outside lower center', ncol=4, frameon=False, fontsize=7)
    return fig


def hilbert_figure(comparisons):
    dims=sorted({r['dimension'] for r in comparisons})
    fig, axes=plt.subplots(1,2,figsize=(7.2,2.8),layout='constrained',sharey=True)
    for sequence, ax in zip(['sobol','halton'],axes):
        for d,color in zip(dims,COLORS):
            values=sorted((r for r in comparisons if r['sequence']==sequence and r['dimension']==d),key=lambda r:r['n'])
            ax.plot([r['n'] for r in values],[r['cpu_over_gpu'] for r in values],color=color,marker='o',markersize=3,label=f'd = {d}')
        axes_style(ax)
        ax.set(title=sequence.title()+' inputs',yscale='log')
        ax.axhline(1,color='#555555',linestyle='--',linewidth=.8)
    axes[0].set_ylabel('JAX CPU / JAX GPU median time')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=5,frameon=False,fontsize=7)
    return fig
