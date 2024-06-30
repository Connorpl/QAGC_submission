import sys
from typing import Any

"""
####################################
add imports here
####################################
"""

from itertools import combinations
from openfermion import QubitOperator, jordan_wigner
from typing import Optional, Union, Tuple, List, Sequence, Mapping
from quri_parts.openfermion.operator import operator_from_openfermion_op
from quri_parts.circuit.transpile import RZSetTranspiler
from quri_parts.core.operator import (
    pauli_label,
    Operator,
    PauliLabel,
    pauli_product,
    PAULI_IDENTITY,
)
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from quri_parts.core.operator.representation import (
    BinarySymplecticVector,
    pauli_label_to_bsv,
    transition_amp_representation,
    transition_amp_comp_basis,
)
from quri_parts.core.state import ComputationalBasisState, ParametricCircuitQuantumState
from quri_parts.qulacs.sampler import create_qulacs_vector_sampler
from openfermion import FermionOperator as FO
import numpy as np
import scipy
import copy
import os
from scipy.sparse import coo_matrix
from random import randint
import xgboost as xgb
os.environ['KMP_DUPLICATE_LIB_OK']='True'

### for Regressor 
from datetime import datetime
import stopit
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

"""
####################################
end imports 
####################################
"""

sys.path.append("../")
from utils.challenge_2024 import ChallengeSampling, problem_hamiltonian

challenge_sampling = ChallengeSampling()

class Solver:
    def __init__(
        self,
        is_classical,
        hamiltonian: Operator,
        pool: list[Operator],
        n_qubits: int,
        n_ele_cas: int,
        sampler,
        iter_max: int = 10,
        sampling_shots: int = 10**4,
        post_selected: bool = True,
        atol: float = 1e-5,
        round_op_config: Union[dict, Tuple[Optional[int], Optional[float]]] = (None, 1e-2),
        num_precise_gradient=None,
        max_num_converged: int = 1,
        final_sampling_shots_coeff: float = 1.0,
        check_duplicate: bool = True,
        reset_ignored_inx_mode: int = 10,
    ):
        self.is_classical = is_classical
        self.hamiltonian: Operator = hamiltonian
        self.pool: list[Operator] = pool
        self.n_qubits: int = n_qubits
        self.n_ele_cas: int = n_ele_cas
        self.iter_max: int = iter_max
        self.sampling_shots: int = sampling_shots
        self.atol: float = atol
        self.sampler = sampler
        self.sv_sampler = create_qulacs_vector_sampler()
        self.post_selected: bool = post_selected
        self.check_duplicate: bool = check_duplicate
        # initialization
        hf_state = ComputationalBasisState(self.n_qubits, bits=2 ** self.n_ele_cas - 1)
        self.hf_state = hf_state
        self.comp_basis = [hf_state]
        # gradient
        if round_op_config is None:
            round_op_config = (None, None)
        num_pickup: int = round_op_config["num_pickup"] if isinstance(round_op_config, dict) else round_op_config[0]
        coeff_cutoff: float = round_op_config["cutoff"] if isinstance(round_op_config, dict) else round_op_config[1]
        self.num_pickup = num_pickup
        self.coeff_cutoff = coeff_cutoff
        round_ham = round_hamiltonian(hamiltonian, num_pickup=num_pickup, coeff_cutoff=coeff_cutoff)
        self.round_hamiltonian = round_ham
        self._is_grad_round: bool = not (num_pickup is None and coeff_cutoff is None)
        self.gradient_pool: List[Operator] = [commutator(round_ham, op) for op in pool]
        self.precise_grad_vals_mem: dict = {}
        self.gradient_vector_history = []
        self.num_precise_gradient: int = len(pool) if num_precise_gradient is None else num_precise_gradient
        self.pauli_rotation_circuit_qsci = PauliRotationCircuit([], [], [], n_qubits)
        self.ignored_gen_inx = []
        self.reset_ignored_inx_mode: int = reset_ignored_inx_mode if reset_ignored_inx_mode > 0 else iter_max
        # convergence
        assert max_num_converged >= 1
        self.final_sampling_shots: int = int(final_sampling_shots_coeff * sampling_shots)
        self.max_num_converged: int = max_num_converged
        self.num_converged: int = 0
        # results
        self.qsci_energy_history: list = []
        self.opt_energy_history: list = []
        self.operator_index_history: list = []
        self.gradient_history: list = []
        self.param_values: list = []
        self.raw_energy_history = []
        self.sampling_results_history = []
        self.comp_basis_history = []
        self.opt_param_value_history = []
        self.generator_history: list = []
        self.generator_qubit_indices_history: list = []

        if self.num_precise_gradient > len(pool):
            self.num_precise_gradient = len(pool)


    
    # Modify ADD VQE
    def _get_optimized_parameter(
        self, vec_qsci: np.ndarray, comp_basis: list[ComputationalBasisState]
    ) -> float:
        generator_qp = self.pool[self.operator_index_history[-1]]
        ham_sparse = generate_truncated_hamiltonian(self.hamiltonian, comp_basis)
        commutator_sparse = generate_truncated_hamiltonian(
            1j * self.precise_grad_vals_mem[self.operator_index_history[-1]], comp_basis
        )
        exp_h = (vec_qsci.T.conj() @ ham_sparse @ vec_qsci).item().real
        exp_commutator = (
            (vec_qsci.T.conj() @ commutator_sparse @ vec_qsci).item().real
        )
        php = generator_qp * self.hamiltonian * generator_qp
        php_sparse = generate_truncated_hamiltonian(php, comp_basis)
        exp_php = (vec_qsci.T.conj() @ php_sparse @ vec_qsci).item().real
        cost_e2 = (
            lambda x: exp_h * np.cos(x[0]) ** 2
            + exp_php * np.sin(x[0]) ** 2
            + exp_commutator * np.cos(x[0]) * np.sin(x[0])
        )
        result_qsci = scipy.optimize.minimize(
            cost_e2, np.array([0.0]), method="BFGS", options={"disp": False, "gtol": 1e-6}
        )
        try:
            assert result_qsci.success
        except:
            print("try optimization again...")
            result_qsci = scipy.optimize.minimize(
                cost_e2, np.array([0.1]), method="BFGS", options={"disp": False, "gtol": 1e-6}
            )
            if not result_qsci.success:
                print("*** Optimization failed, but we continue calculation. ***")
        print(f"θ: {result_qsci.x}")
        return float(result_qsci.x)

    def run(self) -> float:
        vec_qsci, val_qsci = diagonalize_effective_ham(self.hamiltonian, self.comp_basis)
        self.qsci_energy_history.append(val_qsci)
        for itr in range(1, self.iter_max + 1):
            print(f"iteration: {itr}")
            grad_vals = np.zeros(len(self.pool), dtype=float)
            for j, grad in enumerate(self.gradient_pool):
                grad_mat = generate_truncated_hamiltonian(1j * grad, self.comp_basis)
                grad_vals[j] = (vec_qsci.T @ grad_mat @ vec_qsci).real
            sorted_indices = np.argsort(np.abs(grad_vals))[::-1]

            # find largest index of generator
            precise_grad_vals = {}
            if self.num_precise_gradient is not None and self._is_grad_round:
                # calculate the precise value of gradient
                for i_ in list(sorted_indices):
                    if i_ not in self.ignored_gen_inx:
                        if i_ in self.precise_grad_vals_mem.keys():
                            grad = self.precise_grad_vals_mem[i_]
                        else:
                            grad = commutator(self.hamiltonian, self.pool[i_])
                            self.precise_grad_vals_mem[i_] = grad
                        grad_val = (
                            vec_qsci.T
                            @ generate_truncated_hamiltonian(1j * grad, self.comp_basis)
                            @ vec_qsci
                        )
                        precise_grad_vals[i_] = grad_val
                    else:
                        pass
                    if len(precise_grad_vals.keys()) >= self.num_precise_gradient:
                        break
                # print(precise_grad_vals)
                sorted_keys = sorted(precise_grad_vals.keys(), key=lambda x: abs(precise_grad_vals[x]), reverse=True)
                # print(len(sorted_keys),self.num_precise_gradient)
                # assert len(sorted_keys) == self.num_precise_gradient

                # select generator whose abs. gradient is second largest when same generator is selected twice in a row
                if self.check_duplicate:
                    if (len(self.operator_index_history) >= 1 and len(sorted_keys) >= 2) and \
                            (sorted_keys[0] == self.operator_index_history[-1]):
                        largest_index: int = sorted_keys[1]
                        print("selected second largest gradient")
                        self.ignored_gen_inx.append(sorted_keys[0])
                        print(f"index {sorted_keys[0]} added to ignored list")
                    else:
                        largest_index: int = sorted_keys[0]
                else:
                    largest_index = sorted_indices[0]
                grad_vals = precise_grad_vals.values()
                print(
                    f"new generator: {str(self.pool[largest_index]).split('*')}, index: {largest_index} "
                    f"out of {len(self.pool)}. # precise gradient: {self.num_precise_gradient}"
                )
                self.gradient_vector_history.append(key_sortedabsval(precise_grad_vals))
            else:
                largest_index = sorted_indices[0]
            self.operator_index_history.append(largest_index)
            self.gradient_history.append(np.abs(max(grad_vals)))
            operator_coeff_term = str(self.pool[largest_index]).split("*")
            new_coeff, new_pauli_str = float(operator_coeff_term[0]), operator_coeff_term[1]
            self.generator_history.append(new_pauli_str)

            # add new generator to ansatz
            new_param_name = f"theta_{itr}"
            circuit_qsci = self.pauli_rotation_circuit_qsci.add_new_gates(new_pauli_str, new_coeff, new_param_name)
            new_param_value = self._get_optimized_parameter(vec_qsci, self.comp_basis)
            if np.isclose(new_param_value, 0.):
                self.ignored_gen_inx.append(largest_index)
                print(f"index {largest_index} added to ignored list")
            self.opt_param_value_history.append(new_param_value)
            if self.pauli_rotation_circuit_qsci.fusion_mem:
                self.param_values[
                    self.pauli_rotation_circuit_qsci.fusion_mem[0]
                ] += new_param_value
            else:
                if np.isclose(0.0, new_param_value):
                    circuit_qsci = self.pauli_rotation_circuit_qsci.delete_newest_gate()
                else:
                    self.param_values.append(new_param_value)
            try:
                new_gen_indices = sorted(circuit_qsci.gates[-1].target_indices)
            except IndexError:
                print(f"ansatz seems to have no gates since optimized parameter was {new_param_value}")
                raise

            # increase sampling shots when same generator is selected twice in a row or parameter is close to 0.
            is_alert = new_gen_indices in self.generator_qubit_indices_history or np.isclose(0.0, new_param_value)
            self.generator_qubit_indices_history.append(new_gen_indices)
            sampling_shots = self.final_sampling_shots if is_alert else self.sampling_shots

            # prepare circuit for QSCI
            parametric_state_qsci = prepare_parametric_state(self.hf_state, circuit_qsci)
            target_circuit = parametric_state_qsci.parametric_circuit.bind_parameters(self.param_values)
            transpiled_circuit = RZSetTranspiler()(target_circuit)
            
            if self.is_classical:
                counts = self.sv_sampler(transpiled_circuit, shots=sampling_shots)
                pass

            else:
                # QSCI
                try:
                    "Using quantum resources"
                    counts = self.sampler(transpiled_circuit, sampling_shots)
                except ExceededError as e:
                    print(str(e))
                    return min(self.qsci_energy_history)
            self.comp_basis = pick_up_bits_from_counts(
                counts=counts,
                n_qubits=self.n_qubits,
                R_max=num_basis_symmetry_adapted_cisd(self.n_qubits),
                threshold=1e-10,
                post_select=self.post_selected,
                n_ele=self.n_ele_cas,
            )
            self.sampling_results_history.append(counts)
            self.comp_basis_history.append(self.comp_basis)
            vec_qsci, val_qsci = diagonalize_effective_ham(
                self.hamiltonian, self.comp_basis
            )
            self.qsci_energy_history.append(val_qsci)
            # print(f"basis selected: {[bin(b.bits)[2:].zfill(self.n_qubits) for b in self.comp_basis]}")
            print(f"QSCI energy: {val_qsci}, (new generator {new_pauli_str})")

            # terminate condition
            if (
                abs(self.qsci_energy_history[-2] - self.qsci_energy_history[-1])
                < self.atol
            ):
                self.num_converged += 1
                if self.num_converged == self.max_num_converged:
                    break
                else:
                    continue

            # empty ignored index list periodically
            if itr % self.reset_ignored_inx_mode == 0:
                print(f"ignored list emptied: {self.ignored_gen_inx} -> []")
                self.ignored_gen_inx = []
        return min(self.qsci_energy_history)


class PauliRotationCircuit:
    def __init__(
        self, generators: list, coeffs: list, param_names: list, n_qubits: int
    ):
        self.generators: list = generators
        self.coeffs: list = coeffs
        self.param_names: list = param_names
        self.n_qubits: int = n_qubits
        self.fusion_mem: list = []
        self.generetors_history: list = []

    def __call__(self):
        return self.construct_circuit()

    def construct_circuit(
        self, generators=None
    ) -> LinearMappedUnboundParametricQuantumCircuit:
        circuit = LinearMappedUnboundParametricQuantumCircuit(self.n_qubits)
        if generators is None:
            generators = self.generators
        for generator, coeff, name in zip(generators, self.coeffs, self.param_names):
            param_name = circuit.add_parameter(name)
            if isinstance(generator, str):
                generator = pauli_label(generator)
            else:
                raise
            pauli_index_list, pauli_id_list = zip(*generator)
            coeff = coeff.real
            circuit.add_ParametricPauliRotation_gate(
                pauli_index_list,
                pauli_id_list,
                {param_name: -2.0 * coeff},
            )
        return circuit

    def add_new_gates(
        self, generator: str, coeff: float, param_name: str
    ) -> LinearMappedUnboundParametricQuantumCircuit:
        self._reset()
        self.generetors_history.append(generator)
        for i, (g, n) in enumerate(zip(self.generators[::-1], self.param_names[::-1])):
            if is_equivalent(generator, g):
                self.fusion_mem = [-i]
                print(f"FUSED: {g, generator}")
                break
            elif is_commute(generator, g):
                continue
            else:
                break
        if not self.fusion_mem:
            self.generators.append(generator)
            self.coeffs.append(coeff)
            self.param_names.append(param_name)
        return self.construct_circuit()

    def delete_newest_gate(self) -> LinearMappedUnboundParametricQuantumCircuit:
        self._reset()
        self.generators = self.generators[:-1]
        self.coeffs = self.coeffs[:-1]
        self.param_names = self.param_names[:-1]
        return self.construct_circuit()

    def _reset(self):
        self.fusion_mem = []


def diagonalize_effective_ham(
    ham_qp: Operator, comp_bases_qp: list[ComputationalBasisState]
) -> Tuple[np.ndarray, np.ndarray]:
    effective_ham_sparse = generate_truncated_hamiltonian(ham_qp, comp_bases_qp)
    assert np.allclose(effective_ham_sparse.todense().imag, 0)
    effective_ham_sparse = effective_ham_sparse.real
    if effective_ham_sparse.shape[0] > 10:
        eig_qsci, vec_qsci = scipy.sparse.linalg.eigsh(
            effective_ham_sparse, k=1, which="SA"
        )
        eig_qsci = eig_qsci.item()
        vec_qsci = vec_qsci.squeeze()
    else:
        eig_qsci, vec_qsci = np.linalg.eigh(effective_ham_sparse.todense())
        eig_qsci = eig_qsci[0]
        vec_qsci = np.array(vec_qsci[:, 0])

    return vec_qsci, eig_qsci

# Modify
def generate_truncated_hamiltonian(
    hamiltonian: Operator,
    states: Sequence[ComputationalBasisState],
) -> scipy.sparse.spmatrix:
    """Generate truncated Hamiltonian on the given basis states."""
    dim = len(states)
    values = []
    row_ids = []
    column_ids = []
    h_transition_amp_repr = transition_amp_representation(hamiltonian)
    for m in range(dim):
        for n in range(m, dim):
            mn_val = transition_amp_comp_basis(
                h_transition_amp_repr, states[m].bits, states[n].bits
            )
            if mn_val:
                values.append(mn_val)
                row_ids.append(m)
                column_ids.append(n)
                if m != n:
                    values.append(mn_val.conjugate())
                    row_ids.append(n)
                    column_ids.append(m)
    truncated_hamiltonian = coo_matrix(
        (values, (row_ids, column_ids)), shape=(dim, dim)
    ).tocsc(copy=False)
    truncated_hamiltonian.eliminate_zeros()

    return truncated_hamiltonian


def _add_term_from_bsv(
    bsvs: List[List[Tuple[int, int]]], ops: List[Operator]
) -> Operator:
    ret_op = Operator()
    op0_bsv, op1_bsv = bsvs[0], bsvs[1]
    op0, op1 = ops[0], ops[1]
    for i0, (pauli0, coeff0) in enumerate(op0.items()):
        for i1, (pauli1, coeff1) in enumerate(op1.items()):
            bitwise_string = str(
                bin(
                    (op0_bsv[i0][0] & op1_bsv[i1][1])
                    ^ (op0_bsv[i0][1] & op1_bsv[i1][0])
                )
            )
            if bitwise_string.count("1") % 2 == 1:
                pauli_prod_op, pauli_prod_phase = pauli_product(pauli0, pauli1)
                tot_coef = 2 * coeff0 * coeff1 * pauli_prod_phase
                ret_op.add_term(pauli_prod_op, tot_coef)
    return ret_op


def pauli_string_to_bsv(pauli_str: str) -> BinarySymplecticVector:
    return pauli_label_to_bsv(pauli_label(pauli_str))


def get_bsv(pauli: Union[PauliLabel, str]) -> BinarySymplecticVector:
    if isinstance(pauli, str):
        bsv = pauli_string_to_bsv(pauli)
    else:
        bsv = pauli_label_to_bsv(pauli)
    return bsv


def is_commute(pauli1: Union[PauliLabel, str], pauli2: Union[PauliLabel, str]) -> bool:
    bsv1 = get_bsv(pauli1)
    bsv2 = get_bsv(pauli2)
    x1_z2 = bsv1.x & bsv2.z
    z1_x2 = bsv1.z & bsv2.x
    is_bitwise_commute_str = str(bin(x1_z2 ^ z1_x2)).split("b")[-1]
    return sum(int(b) for b in is_bitwise_commute_str) % 2 == 0


def is_equivalent(
    pauli1: Union[PauliLabel, str], pauli2: Union[PauliLabel, str]
) -> bool:
    bsv1 = get_bsv(pauli1)
    bsv2 = get_bsv(pauli2)
    return bsv1.x == bsv2.x and bsv1.z == bsv2.z


def operator_bsv(op: Operator) -> List[Tuple[int, int]]:
    ret = []
    for pauli in op.keys():
        bsv_pauli = get_bsv(pauli)
        ret.append((bsv_pauli.x, bsv_pauli.z))
    return ret


def round_hamiltonian(op: Operator, num_pickup: int = None, coeff_cutoff: float = None):
    ret_op = Operator()
    if coeff_cutoff in [None, 0.0] and num_pickup is None:
        return op
    sorted_pauli = sorted(op.keys(), key=lambda x: abs(op[x]), reverse=True)
    if num_pickup is not None:
        sorted_pauli = sorted_pauli[:num_pickup]
    if coeff_cutoff is None:
        coeff_cutoff = 0
    for pauli in sorted_pauli:
        coeff = op[pauli]
        if abs(coeff) < coeff_cutoff:
            pass
        else:
            ret_op += Operator({pauli: coeff})
    return ret_op


def commutator(
    op0: Union[Operator, float, int, complex], op1: Union[Operator, float, int, complex]
) -> Operator:
    if not isinstance(op0, Operator) or not isinstance(op1, Operator):
        return Operator({PAULI_IDENTITY: 0.0})
    else:
        assert isinstance(op0, Operator) and isinstance(op1, Operator)
        op0_bsv = operator_bsv(op0)
        op1_bsv = operator_bsv(op1)
        ret_op = _add_term_from_bsv([op0_bsv, op1_bsv], [op0, op1])
        return ret_op


def prepare_parametric_state(initial_state, ansatz):
    circuit = LinearMappedUnboundParametricQuantumCircuit(initial_state.qubit_count)
    circuit += initial_state.circuit
    circuit += ansatz
    return ParametricCircuitQuantumState(initial_state.qubit_count, circuit)


def key_sortedabsval(data: Union[list, dict, np.ndarray], round_: int = 5) -> dict:
    if isinstance(data, dict):
        sorted_keys = sorted(data.keys(), key=lambda x: abs(data[x]), reverse=True)
    else:
        sorted_keys = np.argsort(np.abs(data))[::-1]
    ret_dict = {}
    for k in sorted_keys:
        val = float(data[int(k)].real)
        assert np.isclose(val.imag, 0.0)
        ret_dict[int(k)] = round(val, round_)
    return ret_dict

# CHANGE
def create_qubit_adapt_pool_XY_XXXY(
    n_qubits,
    use_singles: bool = False,
    single_excitation_dominant: bool = False,
    double_excitation_dominant: bool = False,
    mode: list[int] = None,
    n_electrons: int = None,
) -> list[Operator]:
    operator_pool_qubit = []
    if use_singles:
        for p, q in combinations(range(n_qubits), 2):
            if single_excitation_dominant and not (p < n_electrons <= q):
                continue
            operator_pool_qubit.append(QubitOperator(f"X{p} Y{q}", 1.0))
    if mode is None:
        mode = [0, 1, 2, 3]
    for m in mode:
        assert m in [0, 1, 2, 3, 4]
        if m == 4:
            mode = [4]
            break
    for p, q, r, s in combinations(range(n_qubits), 4):
        if double_excitation_dominant and not (q < n_electrons <= r):
            continue
        for m in mode:
            x_index = m if m in [0, 1, 2, 3] else randint(0, 3)
            p_list = ["Y" if _ == x_index else "X" for _ in range(4)]
            gen_string_list = " ".join(
                [f"{p}{i}" for p, i in zip(p_list, (p, q, r, s))]
            )
            operator_pool_qubit.append(QubitOperator(gen_string_list, 1.0))
    operator_pool_qubit = [
        operator_from_openfermion_op(op) for op in operator_pool_qubit
    ]
    return operator_pool_qubit


def num_basis_symmetry_adapted_cisd(n_qubits: int):
    return (n_qubits**4 - 4 * n_qubits**3 + 20 * n_qubits**2 + 64) // 64


def pick_up_bits_from_counts(
    counts: Mapping[int, Union[int, float]],
    n_qubits,
    R_max=None,
    threshold=None,
    post_select=False,
    n_ele=None,
):
    sorted_keys = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
    if threshold is None:
        heavy_bits = sorted_keys
    else:
        heavy_bits = [bit for bit in sorted_keys if counts[bit] >= threshold]
    if post_select:
        assert n_ele is not None
        heavy_bits = [i for i in heavy_bits if bin(i).count("1") == n_ele]
    if R_max is not None:
        heavy_bits = heavy_bits[:R_max]
    comp_bases_qp = [
        ComputationalBasisState(n_qubits, bits=int(key)) for key in heavy_bits
    ]
    return comp_bases_qp


class Wrapper:
    def __init__(self, number_qubits, ham, is_classical, use_singles, num_pickup, coeff_cutoff, self_selection, iter_max, sampling_shots, atol, final_sampling_shots_coeff, num_precise_gradient, max_num_converged, reset_ignored_inx_mode) -> None:
        challenge_sampling.reset()

        self.number_qubits = number_qubits
        self.ham = ham
        self.is_classical = is_classical #use SV solver
        self.use_singles = use_singles #include single excitations in operator pool
        self.num_pickup = num_pickup #retain largest N terms in Hamiltonian
        self.coeff_cutoff = coeff_cutoff #cutoff all terms smaller than this from the num_pickup terms remaining
        self.post_selection = self_selection #force it to work in subspace with correctr number of 1s and 0s
        self.iter_max = iter_max #max total iterations
        self.sampling_shots = sampling_shots #how many shots to use per iteration
        self.atol = atol # the tolerance at which we say it is converged
        self.final_sampling_shots_coeff = final_sampling_shots_coeff #how many more shots to use in the calculatino if the same operator appears twice or the operator parameter is close to zero
        self.num_precise_gradient = num_precise_gradient #how many operators from pool to calculate gradient more precisely 
        self.max_num_converged = max_num_converged #how many iterations does it need to stay within atol to be considered converged
        self.reset_ignored_inx_mode = reset_ignored_inx_mode #after how many iterations do we allow previously used operators to be used again

    def result_for_evaluation(self, seed: int, hamiltonian_directory: str) -> tuple[Any, float]:
        energy_final = self.get_result(seed, hamiltonian_directory)
        total_shots = challenge_sampling.total_shots
        return energy_final, total_shots

    def get_result(self, seed: int, hamiltonian_directory: str) -> float:
        """
            param seed: the last letter in the Hamiltonian data file, taking one of the values 0,1,2,3,4
            param hamiltonian_directory: directory where hamiltonian data file exists
            return: calculated energy.
        """
        n_qubits = self.number_qubits
        ham = self.ham
        n_electrons = n_qubits // 2
        use_singles = self.use_singles
        jw_hamiltonian = jordan_wigner(ham)
        qp_hamiltonian = operator_from_openfermion_op(jw_hamiltonian)
        num_pickup, coeff_cutoff = self.num_pickup, self.coeff_cutoff
        post_selection = self.post_selection
        mps_sampler = challenge_sampling.create_sampler()
        pool = create_qubit_adapt_pool_XY_XXXY(
            n_qubits,
            use_singles=use_singles,
            single_excitation_dominant=True,
            double_excitation_dominant=True,
            mode=[4],
            n_electrons=n_electrons,
        )

        solver = Solver(
            self.is_classical,
            qp_hamiltonian,
            pool,
            n_qubits=n_qubits,
            n_ele_cas=n_electrons,
            sampler=mps_sampler,
            iter_max=self.iter_max,
            post_selected=post_selection,
            sampling_shots=self.sampling_shots,
            atol=self.atol,
            final_sampling_shots_coeff=self.final_sampling_shots_coeff,
            round_op_config=(num_pickup, coeff_cutoff),
            num_precise_gradient=self.num_precise_gradient,
            max_num_converged=self.max_num_converged,
            check_duplicate=True,
            reset_ignored_inx_mode=self.reset_ignored_inx_mode,
        )
        res = solver.run()
        return res

### Our solution ###

# Code form answer.py
def round_hamiltonian_inner(op: Operator, num_pickup: int = None, coeff_cutoff: float = None):
    ret_op = Operator()
    if coeff_cutoff in [None, 0.0] and num_pickup is None:
        return op
    sorted_pauli = sorted(op.keys(), key=lambda x: abs(op[x]), reverse=True)
    if num_pickup is not None:
        sorted_pauli = sorted_pauli[:num_pickup]
    if coeff_cutoff is None:
        coeff_cutoff = 0
    for pauli in sorted_pauli:
        coeff = op[pauli]
        if abs(coeff) < coeff_cutoff:
            pass
        else:
            ret_op += Operator({pauli: coeff})
    return ret_op

# stringify single and double digit numbers for filenames
def format_number_str(number: int):
    return number if number > 9 else f"0{number}"

# save the input hamiltonian in 'hamiltonian' folder in '.data' format
def save_hamiltonian(hamiltonian: FO, number_of_qubits: int, seed: int):
    seed_str = format_number_str(seed)
    number_of_qubits_str = format_number_str(number_of_qubits)
    file_name = f"{number_of_qubits_str}qubits_{seed_str}.data"
    print(f"Saving file {file_name}")
    of.utils.save_operator(
        hamiltonian,
        file_name=file_name,
        data_directory="../hamiltonian",
        allow_overwrite=False,
        plain_text=False,
    )

# search through 'hamiltonian' directory for the next available seed
def check_available_seed():
    # assign directory
    directory = "../hamiltonian"
    max_seed = 0

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f) and ".data" in f:
            seed = int(filename[9:11])
            if seed > max_seed:
                max_seed = seed

    return max_seed + 1

# Reads the ham - no further proccessing
def process_file(folder_path, file_name):
    # Load the binary file using NumPy
    file_path = os.path.join(folder_path, file_name)
    # data = np.load(file_path)
    postfix = file_name[-7:]
    ham = problem_hamiltonian(file_name[:2], postfix[:2], folder_path)
    # Return a tuple containing the first two letters of the file name and its content
    return file_name[:2], ham

def flatten(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(float(item))
    return flat_list

# Function to convert ham to a vector
def ham_to_vector(ham):
    vactor = []

    if isinstance(ham, FO):
        # Extract the terms and coefficients
        terms = ham.terms
        vector = flatten(flatten(terms))
        vector = [element for element in vector if not (-1.0 <= element <= 1.0)]
        # pick top 1000 - check on ABS
        sorted_vector = np.sort(vector)
        # Check if the vector has more than 1000 elements
        # if len(sorted_vector) > 3000:
            # Get the first 500 and the last 500 elements
        #     upper = sorted_vector[:900].any()
        #     lower = sorted_vector[-600:].any()
        #     vector = [element for element in vector if (element >=upper or element <=lower)]

        # Iterate over the terms and their coefficients
        for term, coefficient in terms.items():
            vector.append(float(coefficient.real))

        ##  Convert to a numpy array
        vector = [element for element in vector if not (-0.1 <= element <= 0.1)]
        vector = np.array(flatten(vector), order='C', dtype=float)

        # Print to log
        print_to_file(">>>> adding ham of size " + str(len(terms)))

    else:
        typo = type(ham).__name__
        raise TypeError("Expected ham to be a FermionOperator, but got {}".format(typo))

    # Return the flat vector
    return vector

# Make sure all ham are represented using the same number of bytes
def vec_to_fixed_size_vec(size, vec_in):

    # Ensure vec is a numpy array
    vec = np.array(vec_in)
    # Pad with zeros if vec is shorter than size
    if len(vec) < size:
        return np.pad(vec, (0, size - len(vec)), 'constant')

    # Return vec as is if it is already of the correct size
    elif len(vec) > size:
        raise ValueError("Size of X - padding failed. Vec larger than max size: " + str(size))

    else:
        return vec

        ### 3 - is_classical, binary
        ### 4 - use_singles, binary
        ### 5 - num_pickup, int definitely > 1 (probably want it to grow with number of qubits)
        ### 6 - coeff_cutoff, float definitely > 0 and < 1  ( probably <1e-3 )
        ### 7 - self_selection, binary
        ### 8 - iter_max, int definitely > 1 (want large)
        ### 9 - sampling_shots, int definitely >1 probably want fairly large, at least 100
        ### 10- atol, float definitely >0 and < 1, probably < 1e-3
        ### 11- final_sampling_shots_coeff, int definitely > 0 and probably < 10
        ### 12- num_precise_gradient, int definitely >0
        ### 13- max_num_converged, int definitely > 1
        ### 14- reset_ignored_inx_mode, int deffinitely >=0
        ### def __init__(self, number_qubits, ham, is_classical, use_singles, num_pickup, coeff_cutoff, self_selection, iter_max, sampling_shots, atol, final_sampling_shots_coeff, num_precise_gradient, max_num_converged, reset_ignored_inx_mode) -> None:
# Wrapper(n_qubits, ham, False, True, 100, 0.001, False, 100, 10**5, 1e-6, 5, 128,  2, 0)
#            1        2   3      [0]4  5   [0]**6   7     8     9    **10  11  12  13  14
# Random hyper params
def generate_integers():
    random_values_int = []

    ### 4 - use_singles, binary - False never worked
    random_values_int.append(1)

    # num_pickup: 10-100 - IND:5
    ###num_pickup, int definitely > 1 (probably want it to grow with number of qubits)
    random_values_int.append(random.randint(50, 1000))

    ###self_selection, binary  IND:7
    random_values_int.append(random.randint(0, 1))

    # iter_max: 10-100 - IND:8
    ###iter_max, int definitely > 1 (want large)
    random_values_int.append(random.randint(100, 1000000))

    # sampling_shots 10**1 to 10**5 - IND:9
    ###sampling_shots, int definitely >1 probably want fairly large, at least 100
    random_values_int.append(random.randint(100, 10**6))

    # final_sampling_shots_coeff 1 to 10 - IND:11
    ###final_sampling_shots_coeff, int definitely > 0 and probably < 10
    random_values_int.append(random.randint(1, 9))

    # num_precise_gradient 10 to 300 - IND:12
    ###num_precise_gradient, int definitely >0
    random_values_int.append(random.randint(35, 300))

    # max_num_converged 0 to 10 - IND:13
    ###max_num_converged, int definitely > 1
    # assert max_num_converged >= 1
    # 5+ it gives error: IndexError: list index out of range
    random_values_int.append(random.randint(1, 4))

    # reset_ignored_inx_mode 0 to 100 - IND:14
    ##reset_ignored_inx_mode, int deffinitely >=0
    random_values_int.append(random.randint(0, 100))

    # 4, 5, 7, 8, 9, 11, 12, 13, 14
    return random_values_int

# Random hyper params
def generate_floats():
    random_values_float = []

    # coeff_cutoff: 0.01 to 0.0001 - INd:6
    ###coeff_cutoff, float definitely > 0 and < 1  ( probably <1e-3 )
    random_values_float.append(random.uniform(1e-8, 1e-2))

    # atol 1e-2 to 1e-8 ??? ==> the precision/error - IND:10
    ###atol, float definitely >0 and < 1, probably < 1e-3  --> need to move!
    random_values_float.append(random.uniform(1e-8, 1e-3))

    # IND:6 and IND:10
    return random_values_float

def generate_hyper_params_vec(index, n_qubits):
    # Init to default values
    x_vec_params = np.array([float(n_qubits), 1, 1, 100, 0.001, 0, 100, 10**5, 1e-6, 5, 128, 2, 0], dtype=float)
    if index > 1:
        rvi = generate_integers()
        rvf = generate_floats()
        # Constract hyper param fabricated new vector
        # x_vec_params = np.array([float(n_qubits), 1,  rvi[0], rvi[1], rvf[0], rvi[2], rvi[3], rvi[4], rvf[1], rvi[5], rvi[6], rvi[7], rvi[8]], dtype=float)
        x_vec_params = np.array([float(n_qubits),   1,       1, rvi[1], rvi[1], rvi[2], rvi[3], rvi[4], rvf[1], rvi[5], rvi[6], rvi[7], rvi[8]], dtype=float)

    # Retun new fabricated params vector
    return x_vec_params

def get_min_hyper_param_record(score, corpus):
    # Find the index of the minimal score in corpus0_score
    min_score_index = score.index(min(score))

    # Get the corresponding record from corpus0
    return corpus[min_score_index]

def get_y_prediction(x_vec_params, ham28_vec, max_size, model):
    #  Create Y prediction from given X
    x_vec = np.append(x_vec_params, ham28_vec)
    x_vec_fixed_size = vec_to_fixed_size_vec(max_size, x_vec)
    y_pred = model.predict([x_vec_fixed_size])
    if y_pred > 0:
        return 0.0
    else:
        return y_pred[0]

def add_noise(opt_n_qubit, ham28_vec, max_size, model):
    round_corpus = []
    round_scores = []
    corpus1 = []
    corpus1_score = []

    for m in range(1, 3000):  # Loop from 1 to 200
        #  Totally silly random
        x_vec_params = generate_hyper_params_vec(m, opt_n_qubit)
        y_pred = get_y_prediction(x_vec_params, ham28_vec, max_size, model)

        #  Add to temp containers
        round_corpus.append(x_vec_params)
        round_scores.append(y_pred)

    # Select negative only and 10% top
    sorted_scores = sorted(copy.deepcopy(round_scores))  # Deepcopy and sort
    cut_off_score = min(0,sorted_scores[101])

    # Add scores and corpus that are less than the cut_off_score
    for score, corpus_entry in zip(round_scores, round_corpus):
        if score <= cut_off_score:
            corpus1.append(corpus_entry)
            corpus1_score.append(score)

    return corpus1, corpus1_score

def min_corpus(corpus1, corpus1_score):
    # reduce by 50%
    corpus2 = []
    corpus2_score = []

    # Select negative only and 10% top
    sorted_scores = sorted(copy.deepcopy(corpus1_score))  # Deepcopy and sort
    cut_off_score = min(0,sorted_scores[int(len(corpus1_score)/2)])

    for score, corpus_entry in zip(corpus1_score, corpus1):
        if score <= cut_off_score:
            corpus2.append(corpus_entry)
            corpus2_score.append(score)

    return corpus2, corpus2_score

def get_best(corpus1, corpus1_score, num_items):
    corpus2 = []
    if num_items > len(corpus1_score):
        return corpus1

    # Select negative only and 10% top
    sorted_scores = sorted(copy.deepcopy(corpus1_score))  # Deepcopy and sort
    cut_off_score = min(0,sorted_scores[num_items])

    # Get best min. score items
    for score, corpus_entry in zip(corpus1_score, corpus1):
        if score <= cut_off_score:
            corpus2.append(corpus_entry)

    # Make sure we are not returning anything too small
    if num_items > len(corpus2):
        return corpus1
    return corpus2


def combiner(item1, item2, method):

    # Ensure the lists are of equal length
    if len(item1) != len(item2):
        raise ValueError("Both lists must have the same length.")

    n = len(item1)
    item3 = []

    if method == 'random':
        item3 = [random.choice([a, b]) for a, b in zip(item1, item2)]

    elif method == 'min':
        item3 = [min(a, b) for a, b in zip(item1, item2)]

    elif method == 'max':
        item3 = [max(a, b) for a, b in zip(item1, item2)]

    elif method == 'cut_half':
        item3 = np.concatenate((item1[:n//2], item2[n//2:]))

    elif method == 'average':
        item3 = [(a + b) / 2 for a, b in zip(item1, item2)]

    else:
        raise ValueError("Invalid method. Choose from 'random', 'min', 'max', 'cut_half', or 'average'.")

    # Ensure the result is of equal length
    if len(item3) != n:
        raise ValueError("Result must have the same length.")

    return item3

def print_to_file(message):
    with open("logger.txt", 'a') as f:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{current_time} - {message}\n")

def regressor(opt_n_qubit, opt_seed, folder_path):
    # Hyperparameters of the quantum init stage - init the return vector and ham28 to predict
    ret_opt = [False, True, 100, 0.001, False, 100, 10**5, 1e-6, 5, 128, 2, 0]
    ham28 = problem_hamiltonian(opt_n_qubit, opt_seed, folder_path)
    ham28_vec = ham_to_vector(ham28)

    # Initialize lists to store X and Y data
    X = [] # Random wrappers created from hamiltonians in folder
    Y = [] # Energy level computed classically for each wrapper in X

    # List all files in the folder
    files = os.listdir(folder_path)
    # Iterate over each file and print its name
    for file_name in files:
        result = process_file(folder_path, file_name)
        n_qubits=int(result[0])
        print_to_file(">> Start processing: "+file_name+" with qubits "+str(n_qubits))
        if n_qubits < 17: # anything less than 20 qubits
            # Get the ham flatten once
            ham=result[1] # Need to load from Elena's files
            ham_vec = ham_to_vector(ham)

            # Compress to wrapper only if too big
            orig=str(len(ham.terms))
            ham.compress(0.05)
            print_to_file(">>> compress to " + str(len(ham.terms)) + " from " + orig)

            # Add some random value with the wrapper
            for i in range(1, 20):  # Loop from 1 to 20
                print_to_file(">>>> Collecting data with qubits "+str(n_qubits) + " iteration " + str(i))
                x_vec_params = generate_hyper_params_vec(i, n_qubits)
                y_vec=0

                try:
                    # wrapper=Wrapper(n_qubits, ham, True, True, 100, 0.001, False, 100, 10**5, 1e-6, 5, 128, 2, 0)
                    param_bool_1 = True if round(x_vec_params[2]) != 0 else False
                    param_bool_2 = True if round(x_vec_params[5]) != 0 else False

                    # Call the wrapper
                    with stopit.ThreadingTimeout(120) as context_manager:
                        wrapper=Wrapper(n_qubits, ham, True, param_bool_1, round(x_vec_params[3]), x_vec_params[4], param_bool_2,
                            round(x_vec_params[6]), round(x_vec_params[7]), x_vec_params[8],
                            round(x_vec_params[9]), round(x_vec_params[10]), round(x_vec_params[11]), round(x_vec_params[12]))
                        y_vec=wrapper.get_result(seed=0, hamiltonian_directory="../hamiltonian")
                    # Did code finish running in under 180 seconds?
                    if context_manager.state == context_manager.TIMED_OUT:
                       y_vec=3.0
                       print_to_file("Y is 0 due to timeout")

                except Exception as e:
                    y_vec=0.0
                    print_to_file("Y is 0 due to exception")

                # Add data to X and Y sets
                if not (y_vec == 0.0):
                    x_vec = np.append(x_vec_params, ham_vec)
                    X.append(x_vec)  # Parameters and ham as a vector
                    Y.append(y_vec)  # E level classically
                    print_to_file(">>>> Register data, X size of " + str(len(x_vec)) + " with energy level " + str(y_vec))
                else:
                    print_to_file(">>>> Skip data with energy level " + str(y_vec))

        print_to_file(">> End Collecting Data with sample "+file_name)
        # x_train = Wrapper(n_qubits, ham, False, True, 100, 0.001, False, 100, 10**5, 1e-6, 5, 128, 2, 0)
        # y_train = Energy level (computed classically) # We make a BIG assumption that the regressor will be able to generalise without retraining on 28 qubits
    # End of loop

    print_to_file(">> Start training Phase. With " + str(len(X)) + " training data")
    if len(X) != len(Y):
        print("size of X: " + str(len(X)))
        print("size of Y: " + str(len(Y)))
        raise ValueError("Size of X and Y must be equal")

    # Calculate the size of each vector in X
    # Find the maximum size among all vectors + 13 for the hyper params vector
    max_size = max(len(ham28_vec), max([len(vector) for vector in X])) + 13

    # Apply vec_to_fixed_size_vec to all vectors in X
    for i in range(len(X)):
        size_before = len(X[i])
        X[i] = vec_to_fixed_size_vec(max_size, X[i])
        size_after = len(X[i])
        if size_after != max_size:
            print("Size before and after: " + size_before + "," + size_after + " with max size:" + str(max_size))
            raise ValueError("Size of X - padding failed. Itr: " + str(i))

    # Split dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    # Parameters
    params = {
        'objective': 'reg:pseudohubererror',
        'max_depth': 5,
        'learning_rate': 0.04,
        'n_estimators': 55,     # Number of trees to fit
        'alpha': 0.1,           # L1 regularization term on weights
        'lambda': 0.1           # L2 regularization term on weights
    }
    chunk_size=9
    num_chunks=len(X)//chunk_size
    print (">>> Train the model with size ", str(chunk_size))

    # Initialize and train the regressor model
    model = xgb.XGBRegressor(**params, random_state=42)
    # Train
    for chunk in range(num_chunks):
        start_idx = chunk * chunk_size
        end_idx = (chunk + 1) * chunk_size
        x_chunk = X[start_idx:end_idx]
        y_chunk = Y[start_idx:end_idx]

        print_to_file(">>>> Start training chunk " + str(chunk))
        if chunk == 0:
            model.fit(x_chunk, y_chunk)
        else:
            model.n_estimators += params['n_estimators']
            model.fit(x_chunk, y_chunk, xgb_model=model)
    # model.fit(x_train,y_train)
    # Evaluate on test data
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print(">>>>>>>>>>>>> Mean Squared Error:", mse, ":: Predictions:", y_pred, ":: True Values:", y_test)

    # Try to predict
    print("Prediction on 28 qubits:")
    print_to_file("Prediction on 28 qubits")

    # Constract hyper param fabricated new vector - generate initial seeds
    corpus0 = []
    corpus0_score = []
    for i in range(1, 200):  # Loop from 1 to 200
        x_vec_params = generate_hyper_params_vec(i, opt_n_qubit)
        y_pred = get_y_prediction(x_vec_params, ham28_vec, max_size, model)

        # Add the corpus0
        corpus0.append(x_vec_params)
        corpus0_score.append(y_pred)

    # opt.
    corpus1 = copy.deepcopy(corpus0)
    corpus1_score = copy.deepcopy(corpus0_score)
    for r in range(1, 50):  # Loop from 1 to 200
        # combiner
        best_res = get_best(corpus1, corpus1_score, 20)
        for m in range(1, 5):  # Loop from 1 to 200
            # 2-input Combiner
            item1, item2 = random.sample(best_res, 2)
            # Mutate
            methods = ['random','min', 'max', 'cut_half', 'average']
            new_item = combiner(item1, item2, random.choice(methods))  # Randomly choose a method to combine parameters
            # Get score
            y_pred = get_y_prediction(new_item, ham28_vec, max_size, model)
            # Add the corpus0
            corpus1.append(x_vec_params)
            corpus1_score.append(y_pred)

        # Add noise every 5 iterations
        if r % 5 == 0:
            # reduce by 50%
            min_res = min_corpus(corpus1, corpus1_score)
            corpus1 = min_res[0]
            corpus1_score = min_res[1]

            # Add a bit of noise
            nosie_res = add_noise(opt_n_qubit, ham28_vec, max_size, model)
            corpus1.extend(nosie_res[0])
            corpus1_score.extend(nosie_res[1])

    # Min.
    ret_opt = get_min_hyper_param_record(corpus1_score, corpus1)

    # return the optimised guess
    return ret_opt
###


class RunAlgorithm:
    def __init__(self) -> None:
        challenge_sampling.reset()

    def result_for_evaluation(self, seed: int, hamiltonian_directory: str) -> tuple[Any, float]:
        energy_final = self.get_result(seed, hamiltonian_directory)
        total_shots = challenge_sampling.total_shots
        return energy_final, total_shots

    def regressor(opt_n_qubit, opt_seed, folder_path):
        ret_opt = [True, True, 100, 0.001, False, 100, 10**5, 1e-6, 5, 128, 2, 0]
        return ret_opt
    
    def get_result(self, seed: int, hamiltonian_directory: str) -> float:
        """
            param seed: the last letter in the Hamiltonian data file, taking one of the values 0,1,2,3,4
            param hamiltonian_directory: directory where hamiltonian data file exists
            return: calculated energy.
        """
        n_qubits = 28
        ham = problem_hamiltonian(n_qubits, seed, hamiltonian_directory)
        ###is_classical, binary
        ###use_singles, binary
        ###num_pickup, int definitely > 1 (probably want it to grow with number of qubits)
        ###coeff_cutoff, float definitely > 0 and < 1  ( probably <1e-3 )
        ###self_selection, binary
        ###iter_max, int definitely > 1 (want large)
        ###sampling_shots, int definitely >1 probably want fairly large, at least 100 
        ###atol, float definitely >0 and < 1, probably < 1e-3 
        ###final_sampling_shots_coeff, int definitely > 0 and probably < 10
        ###num_precise_gradient, int definitely >0 
        ###max_num_converged, int definitely > 1
        ###reset_ignored_inx_mode, int definitely >=0

        """
        ####################################
        add codes here
        ####################################
        """

        # Get better hyper-params
        res_opt = regressor(n_qubits, seed, hamiltonian_directory)
        # The call to the VQE
        param_bool_2 = True if round(res_opt[5]) != 0 else False
        wrapper=Wrapper(n_qubits, ham, True, True, round(res_opt[3]), res_opt[4], param_bool_2,
                      # n_qubits, ham, True, True, 100,               0.001,      False,  
                            round(res_opt[6]), round(res_opt[7]), min([res_opt[8],1e-7]),
                      #     100,               10**5,             1e-6,  
                            round(res_opt[9]), round(res_opt[10]), 4, round(res_opt[12]))
                      #     5,                 128,                2,                  0
        
        # wrapper = Wrapper(n_qubits, ham, True, res_opt[1], res_opt[2], res_opt[3], res_opt[4], res_opt[5], res_opt[6], res_opt[7], 
        # res_opt[8], res_opt[9], res_opt[10], res_opt[11])
        # wrapper = Wrapper(n_qubits, ham, True, True, 100, 0.001, False, 100, 10**5, 1e-6, 5, 128, 2, 0)
        res=wrapper.get_result(seed=0, hamiltonian_directory="../hamiltonian")
        print("type: ",type(res))

        return res

if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result(seed=0, hamiltonian_directory="../hamiltonian"))
