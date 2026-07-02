import numpy as np
import scipy.sparse as sp
import tqdm


class SKQD:
    def __init__(
        self,
        hamiltonian: sp.csr_matrix,
        num_steps: int,
        t: float,
        initial_state: int = None,
    ):
        self.H = hamiltonian
        self.num_steps = num_steps
        self.t = t
        self.n = self.H.shape[0]

        self.samples_per_step = self.n // self.num_steps
        self.leftover = self.n % self.num_steps

        if self.leftover > 0:
            print(
                f"Warning: H.shape[0] is not perfectly divisible by num_steps. "
                f"Last {self.leftover} samples will be weird."
            )

        self.U = sp.linalg.expm(-1j * self.H * self.t)

        # Initial wavefunction
        self.psi = np.zeros(self.n, dtype=np.complex128)
        if initial_state is None:
            self.psi[:] = 1.0 / np.sqrt(self.n)
            self.initial_state = None
            self.samples = [[]]
        else:
            self.psi[initial_state] = 1.0
            self.initial_state = initial_state
            self.samples = [[initial_state]]

        # Store amplitudes like BARK
        self.amplitudes = [self.psi.copy()]

        # Track which indices are still available
        self.mask = np.ones(self.n, dtype=bool)
        if initial_state is not None:
            self.mask[initial_state] = False

        self.current_step = 0

    def step(self):
        """
        Perform one SKQD step:
        - evolve the state with U
        - sample up to samples_per_step new indices
        - append cumulative sample list to self.samples

        Returns
        -------
        finished : bool
            True if all indices have been sampled or all planned steps are done.
        """
        if self.current_step >= self.num_steps:
            return True

        # Evolve state
        self.psi = self.U @ self.psi
        self.amplitudes.append(self.psi.copy())

        probabilities = np.abs(self.psi) ** 2
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)

        new_indices = []
        i = 0
        while i < self.samples_per_step:
            leftover_indices = np.where(self.mask)[0]
            if leftover_indices.size == 0:
                break

            probabilities_here = probabilities[self.mask]
            prob_sum = probabilities_here.sum()

            if prob_sum <= 0 or not np.isfinite(prob_sum):
                probabilities_here = np.ones_like(probabilities_here) / len(probabilities_here)
            else:
                probabilities_here = probabilities_here / prob_sum

            sampled_index = np.random.choice(leftover_indices, p=probabilities_here)

            new_indices.append(sampled_index)
            self.mask[sampled_index] = False
            i += 1

        self.samples.append(self.samples[-1] + new_indices)
        self.current_step += 1

        finished = self.current_step >= self.num_steps or len(self.samples[-1]) >= self.n
        return finished

    def run(self, progress_bar: bool = True):
        """
        Run SKQD until num_steps are completed, then fill leftover indices one by one
        so that self.samples behaves similarly to BARK.
        """
        iterator = range(self.num_steps)
        if progress_bar:
            iterator = tqdm.tqdm(iterator)

        for _ in iterator:
            finished = self.step()
            if finished:
                break

        # Fill any leftover indices one-by-one, appending intermediate cumulative samples
        while len(self.samples[-1]) < self.n:
            leftover_indices = np.where(self.mask)[0]
            if leftover_indices.size == 0:
                break

            sampled_index = np.random.choice(leftover_indices)
            self.mask[sampled_index] = False
            self.samples.append(self.samples[-1] + [sampled_index])

        return self.samples