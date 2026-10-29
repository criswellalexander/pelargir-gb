'''

Typeset the formulas of docs/flax_flow_architecture.svg. The plain-text source,
flax_flow_architecture_src.svg, is the one to edit; every <text> in it whose content is a key of
FORMULAS is replaced by an outlined <path> of the mathtext string (STIX math; Liberation Sans for
the words, the font browsers substitute for Helvetica), at the same position, size, anchor, weight
and colour, with the plain text kept as the path's <title>. The output needs no fonts to display.

Usage (from the repository root; matplotlib and the Liberation Sans fonts at FONTS are required):
    python docs/typeset_flax_flow_architecture.py [src.svg out.svg]

'''
import os
import re
import sys
import html

from matplotlib.textpath import TextPath, text_to_path
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.text import Text

HERE = os.path.dirname(os.path.abspath(__file__))
## Liberation Sans is metric-compatible with Helvetica (Debian/Ubuntu: fonts-liberation)
FONTS = {'normal': '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
         'bold': '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'}

## plain-text content of the source figure -> mathtext
FORMULAS = {
    'pelargir flax flow emulator: one band flow (D = 6: N_res of the band + x = log10(S_gw + C) in its 5 bins)':
        r'pelargir flax flow emulator: one band flow',
    'N_res (band total)': r'$N_\mathrm{res}$ (band total)',
    'u ~ U(0, 1)': r'$u \sim \mathcal{U}(0, 1)$',
    'GL nodes u_k in log_prob': r'GL nodes $u_k$ in log_prob',
    'S_gw (5 bins)': r'$S_\mathrm{gw}$ (5 bins)',
    'foreground PSD [1/Hz], &#8805; 0': r'foreground PSD [Hz$^{-1}$], $\geq 0$',
    'v ~ U(0, 1) per bin': r'$v \sim \mathcal{U}(0, 1)$ per bin',
    'log10(N + u)': r'$\log_{10}(N + u)$',
    'x = log10(S_gw + C)': r'$x = \log_{10}(S_\mathrm{gw} + C)$',
    'C = 3.645e-41 /Hz (noise floor)': r'$C = 3.645 \times 10^{-41}$ Hz$^{-1}$ (noise floor)',
    'bins with S_gw = 0': r'bins with $S_\mathrm{gw} = 0$',
    'x = log10 C - v log10 2': r'$x = \log_{10} C - v\,\log_{10} 2$',
    '(log10(N+u), x1..x5)': r'$(\log_{10}(N + u),\, x_1, \ldots, x_5)$',
    '(f - median) / half-range': r'($f$ $-$ median) / half-range',
    "every step's log|Jacobian| is added to log_prob": r"every step's $\log|\mathrm{Jacobian}|$ is added to log_prob",
    'zero bins: log|dy/dv| (the slab measure), so the mass below log10 C is P(S_gw = 0)':
        r'zero bins: $\log|\partial y / \partial v|$ (the slab measure), so the mass below $\log_{10} C$ is $P(S_\mathrm{gw} = 0)$',
    'x B / max|z|   (B = 4)': r'$\times\, B / \max|z|$   ($B = 4$)',
    'training data fill [-B, B]': r'training data fill $[-B, B]$',
    'y': r'$y$',
    'in [-B, B]^6': r'in $[-B, B]^6$',
    'z ~ N(0, I_6)': r'$z \sim \mathcal{N}(0, I_6)$',
    'c = (context - median) / half-range': r'$c$ = (context $-$ median) / half-range',
    'Inside one coupling layer (mask m alternates between layers)': r'Inside one coupling layer (mask $m$ alternates between layers)',
    'y_masked = y * m': r'$y_\mathrm{masked} = y \odot m$',
    '[ y_masked || c ]': r'$[\, y_\mathrm{masked} \;\Vert\; c \,]$',
    'Linear 14 to H, ReLU': r'Linear $14 \to H$, ReLU',
    'H = 512 (default)': r'$H = 512$ (default)',
    'Linear H to H, ReLU': r'Linear $H \to H$, ReLU',
    'Linear H to 6 x 25': r'Linear $H \to 6 \times 25$',
    'RQS on [-B-1, B+1] = [-5, 5]': r'RQS on $[-B-1,\, B+1] = [-5, 5]$',
    'applied to the dims with m = 0': r'applied to the dims with $m = 0$',
    'Dims with m = 1 pass through unchanged and condition the spline; outside [-5, 5] the spline is the identity.':
        r'Dims with $m = 1$ pass through unchanged and condition the spline; outside $[-5, 5]$ the spline is the identity.',
    "The Jacobian is triangular, so log|det| is the sum of the spline slopes' logs over the transformed dims.":
        r"The Jacobian is triangular, so $\log|\det|$ is the sum of the spline slopes' logs over the transformed dims.",
    'log P(N_res, S_gw | context) =': r'$\log P(N_\mathrm{res}, S_\mathrm{gw} \mid \mathrm{context}) =$',
    'log sum_k w_k q(y_k | c) |dy_k/d(N, S)|,': r'$\log {\sum}_k w_k\, q(y_k \mid c)\, |\partial y_k / \partial(N, S)|,$',
    'y_k = y(N + u_k, S); 32 Gauss-Legendre u_k': r'$y_k = y(N + u_k, S)$',
    'log_prob runs y to z (inverse spline chain);': r'log_prob runs $y \to z$ (inverse spline chain);',
    'sampling runs z to y and undoes the transforms:': r'sampling runs $z \to y$ and undoes the transforms:',
    'N_res = floor(N + u), S_gw = max(10^x - C, 0)': r'$N_\mathrm{res} = \lfloor N + u \rfloor,\ S_\mathrm{gw} = \max(10^x - C,\, 0)$',
}


def svg_path(s, size, weight):
    '''

    Outline a mathtext string as SVG path data (y down, baseline at 0) and its advance width.

    Arguments
    ------------------------------
    s : str
    	mathtext string.
    size : float
    	Font size [px].
    weight : str
    	'normal' or 'bold' (the words; math keeps its own weights).

    Returns
    ------------------------------
    d : str
    	SVG path data.
    width : float
    	Advance width [px].

    '''
    prop = FontProperties(fname=FONTS[weight], size=size, math_fontfamily='stix')
    s_parsed, ismath = Text()._preprocess_math(s)
    width = text_to_path.get_text_width_height_descent(s_parsed, prop, ismath=ismath)[0]
    out = []
    for verts, code in TextPath((0, 0), s, prop=prop).iter_segments(simplify=False, curves=True):
        pts = ' '.join('{:.2f} {:.2f}'.format(x, -y) for x, y in verts.reshape(-1, 2))
        out.append({Path.MOVETO: 'M', Path.LINETO: 'L', Path.CURVE3: 'Q', Path.CURVE4: 'C'}.get(code, 'Z')
                   + ('' if code == Path.CLOSEPOLY else pts))
    return ''.join(out), width


def main(src=os.path.join(HERE, 'flax_flow_architecture_src.svg'), dst=os.path.join(HERE, 'flax_flow_architecture.svg')):
    '''

    Write dst: src with every FORMULAS text typeset as an outlined path.

    Arguments
    ------------------------------
    src : str
    	The plain-text figure. (Default: docs/flax_flow_architecture_src.svg)
    dst : str
    	The typeset figure. (Default: docs/flax_flow_architecture.svg)

    Returns
    ------------------------------
    None

    '''
    svg = open(src).read()
    used = set()

    def replace(m):
        attrs, content = m.group(1), m.group(2)
        if content not in FORMULAS:
            return m.group(0)
        used.add(content)
        a = dict(re.findall(r'([\w-]+)="([^"]*)"', attrs))
        size = float(a.get('font-size', 13))
        d, w = svg_path(FORMULAS[content], size, 'bold' if a.get('font-weight') == 'bold' else 'normal')
        x0 = float(a['x']) - {'middle': w/2, 'end': w}.get(a.get('text-anchor', 'start'), 0.0)
        return '<path transform="translate({:.2f} {})" fill="{}" d="{}"><title>{}</title></path>'.format(
            x0, a['y'], a.get('fill', '#111827'), d, html.escape(html.unescape(content)))

    svg = re.sub(r'<text ([^>]*)>(.*?)</text>', replace, svg)
    missing = set(FORMULAS) - used
    if missing:
        sys.exit("not found in {}: {}".format(src, sorted(missing)))
    open(dst, 'w').write(svg)


if __name__ == '__main__':
    main(*sys.argv[1:3])
