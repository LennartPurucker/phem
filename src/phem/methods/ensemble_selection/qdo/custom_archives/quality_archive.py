from __future__ import annotations

import numpy as np
from ribs.archives._archive_base import ArchiveBase


class QualityArchive(ArchiveBase):
    """A baseline archive that is not a typical QDO archive.

    It functions like a pure quality population. Thus, allowing us to compare diversity inspired populations of elites
    to simple quality archives. It is a collection of N items w.r.t. some simple characteristics.
    Usually, it is the top N  performing solutions.

    We assume that we want to maximize the objective following the default QDO implementation.

    Parameters
    ----------
    archive_size: int
        Size of the beam.
    behavior_n_dim: int
        Number of dimensions of the behavior space. Needed to correctly store optional behavior values.
    seed: int
        Seed for the random number generator of archive base. (Beam is not random)
    """

    def __init__(self, archive_size, behavior_n_dim, seed, show_analysis=False):
        self._dims = [archive_size]
        self.behavior_n_dim = behavior_n_dim
        self.archive_size = archive_size
        self.stored_values_count = 0

        ArchiveBase.__init__(
            self,
            storage_dims=tuple(self._dims),
            behavior_dim=self.behavior_n_dim,
            seed=seed,
        )
        self._show_analysis = show_analysis
        self._elites_over_time = []

    def get_index(self, behavior_values):
        """Return the index of the position in the archive that should be replaced by the behavior values."""
        # Fill empty bin
        if len(self._occupied_indices) < self.archive_size:
            return tuple(np.unravel_index(np.argmin(self._occupied), self._occupied.shape))

        # Find bin with the smallest value and return (local competition will resolve the if it is inserted or not)
        return tuple(
            np.unravel_index(np.argmin(self._objective_values), self._objective_values.shape)
        )

    def add(self, solution, objective_value, behavior_values, metadata=None):
        status, value = ArchiveBase.add(self, solution, objective_value, behavior_values, metadata)
        self.stored_values_count += 1

        if self._show_analysis and (self.stored_values_count % 50 == 0):
            self._elites_over_time.append([(e.obj, *e.beh) for e in self])

        return status, value
