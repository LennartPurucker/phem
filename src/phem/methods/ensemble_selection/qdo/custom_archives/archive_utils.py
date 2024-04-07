from __future__ import annotations

import numpy as np
from ribs.archives._archive_base import ArchiveBase


def get_quantile_mask(values_to_mask, quantile):
    """Remarks:
    We assume higher is better for the value. Thus, 0.75 -> gives us the first quantile w.r.t. better.
    """
    min_perf_val = np.quantile(values_to_mask, quantile)
    return values_to_mask >= min_perf_val


def remap_clean_up(self, reset_bounds=True):
    """Remap clean up code:

    * Reset the upper and lower bound
    * re-adds all solutions (From buffer and last elites). The last added solution will be the solution with which
        add() was called.

    """
    if reset_bounds:
        self._lower_bounds = np.array([bound[0] for bound in self._boundaries])
        self._upper_bounds = np.array(
            [bound[dim] for bound, dim in zip(self._boundaries, self._dims, strict=False)]
        )

    old_sols = self._solutions[self._occupied_indices_cols].copy()
    old_objs = self._objective_values[self._occupied_indices_cols].copy()
    old_behs = self._behavior_values[self._occupied_indices_cols].copy()
    old_metas = self._metadata[self._occupied_indices_cols].copy()

    self.clear()
    for sol, obj, beh, meta in zip(old_sols, old_objs, old_behs, old_metas, strict=False):
        # Add solutions from old archive.
        status, value = ArchiveBase.add(self, sol, obj, beh, meta)
    for sol, obj, beh, meta in self._buffer:
        # Add solutions from buffer.
        status, value = ArchiveBase.add(self, sol, obj, beh, meta)

    if not self._first_remap_done:
        self._first_remap_done = True
        self._remap_frequency = self._default_remap_frequency

    return status, value
