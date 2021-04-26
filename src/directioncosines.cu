/* -------------------------------------------------------------------------
  Copyright (C) 2016-2017  Miguel Carcamo, Pablo Roman, Simon Casassus,
  Victor Moral, Fernando Rannou - miguel.carcamo@usach.cl

  This program includes Numerical Recipes (NR) based routines whose
  copyright is held by the NR authors. If NR routines are included,
  you are required to comply with the licensing set forth there.

	Part of the program also relies on an an ANSI C library for multi-stream
	random number generation from the related Prentice-Hall textbook
	Discrete-Event Simulation: A First Course by Steve Park and Larry Leemis,
  for more information please contact leemis@math.wm.edu

  For the original parts of this code, the following license applies:

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
* -------------------------------------------------------------------------
*/

#include "directioncosines.cuh"
/**
* Convert (ra,dec) to direction cosines (l,m) relative to
* phase-tracking center (ra0, dec0). All in radians.
* Reference: Synthesis Imaging in Radio Astronomy II, p.388.
 */
__host__ void direccos(double ra, double dec, double ra0, double dec0, double* l, double* m)
{
  double delta_ra = ra - ra0;
  double cosdec = cos(dec);
  *l = cosdec * sin(delta_ra);
  *m = sin(dec) * cos(dec0) - cosdec * sin(dec0) * cos(delta_ra);
}
