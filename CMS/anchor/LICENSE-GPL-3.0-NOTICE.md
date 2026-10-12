# Licence of this directory

Everything under `CMS/anchor/` is distributed under the GNU General Public License, version 3 or (at your option)
any later version (SPDX: `GPL-3.0-or-later`). The full licence text is in [`COPYING`](COPYING); every script here
carries the header `SPDX-License-Identifier: GPL-3.0-or-later`. The rest of the repository is MIT (see `../../LICENSE`).

Why: the first-principles workflow in this directory drives Quantum ESPRESSO and BoltzTraP2.
`boltztrap2_patch/apply_patch.py` is a derivative of BoltzTraP2 (it reproduces and modifies excerpts of BoltzTraP2
24.7.2, Copyright (C) 2017-2024 G. K. H. Madsen, J. Carrete, M. J. Verstraete and co-authors), and
`extract_transport.py` and the SOC tools import BoltzTraP2 at run time. BoltzTraP2 is distributed under
GPL-3.0-or-later, so the whole directory is offered under the same terms rather than drawing the boundary file by file.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
