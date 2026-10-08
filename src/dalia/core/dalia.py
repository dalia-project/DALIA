# Copyright 2024-2025 DALIA authors. All rights reserved.

import logging

from scipy import optimize
from tabulate import tabulate
import copy

from dalia import ArrayLike, NDArray, backend_flags, comm_rank, comm_size, sp, xp
from dalia.configs.dalia_config import DaliaConfig
from dalia.core.model import Model
from dalia.solvers import (
    DenseSolver,
    DistSerinvSolver,
    SerinvSolver,
    SparseSolver,
    STilesSolver,
)
from dalia.utils import (
    DummyCommunicator,
    add_str_header,
    allreduce,
    ascii_logo,
    boxify,
    extract_diagonal,
    format_size,
    free_unused_gpu_memory,
    get_device,
    get_host,
    memory_report,
    print_msg,
    set_device,
    smartsplit,
    synchronize,
    synchronize_gpu,
    check_vector_consistency,
    bcast,
)

if backend_flags["mpi_avail"]:
    from mpi4py import MPI

if backend_flags["nccl_avail"]:
    from cupy.cuda import nccl
else:
    nccl = None

import time

xp.set_printoptions(precision=8, suppress=True, linewidth=150)


class DALIA:
    """DALIA is a Python implementation of the Integrated Nested
    Laplace Approximation (INLA) method.
    """

    def __init__(
        self,
        model: Model,
        config: DaliaConfig,
    ) -> None:
        """Initializes the DALIA object.

        Parameters
        ----------
        model : Model
            Model object from the DALIA Models library.
        config : DaliaConfig
            Configuration object for the DALIA solver.

        Returns
        -------
        None
        """
        # --- Initialize model
        self.model = model

        # --- Initialize DALIA
        self.config = config

        self.inner_iteration_max_iter = self.config.inner_iteration_max_iter
        self.eps_inner_iteration = self.config.eps_inner_iteration
        self.eps_gradient_f = self.config.eps_gradient_f
        self.eps_hessian_f = self.config.eps_hessian_f

        self.verbosity = self.config.verbosity

        # --- Configure HPC
        set_device(comm_rank, comm_size)

        self.n_f_evaluations = 2 * self.model.n_hyperparameters + 1

        # Create the appropriate communicators
        min_q_parallel = 1
        min_solver_size = self.config.solver.min_processes

        if self.model.is_likelihood_gaussian():
            self.n_qeval = 2
        else:
            self.n_qeval = 1

        if backend_flags["mpi_avail"]:
            self.initial_comm_world = MPI.COMM_WORLD

            self.comm_world, self.comm_feval, self.color_feval = smartsplit(
                comm=self.initial_comm_world,
                n_parallelizable_evaluations=self.n_f_evaluations,
                tag="feval",
                min_group_size=min_solver_size * min_q_parallel,
            )
            self.world_size = self.comm_world.size

            self.qeval_world, self.comm_qeval, self.color_qeval = smartsplit(
                comm=self.comm_feval,
                n_parallelizable_evaluations=self.n_qeval,
                tag="qeval",
                min_group_size=min_solver_size,
            )
        else:
            self.initial_comm_world = DummyCommunicator()

            self.comm_world = DummyCommunicator()
            self.comm_feval = DummyCommunicator()
            self.comm_qeval = DummyCommunicator()
            self.qeval_world = DummyCommunicator()

            self.color_feval = 0
            self.color_qeval = 0

            self.world_size = 1

        free_unused_gpu_memory()

        # --- Initialize solver
        if self.config.solver.type == "dense":
            self.solver = DenseSolver(
                config=self.config.solver,
                n=self.model.n_latent_parameters,
            )
        elif self.config.solver.type == "scipy":
            self.solver = SparseSolver(
                config=self.config.solver,
            )
        elif self.config.solver.type == "stiles":
            if self.comm_qeval.size > 1:
                raise ValueError(
                    "The sTiles solver is a shared-memory solver: it cannot be "
                    "distributed over several processes (solver.min_processes > 1)."
                )
            self.solver = STilesSolver(
                config=self.config.solver,
            )
        elif self.config.solver.type == "serinv":
            serinv_parameters = model.get_solver_parameters()
            diagonal_blocksize: int = serinv_parameters["diagonal_blocksize"]
            arrowhead_blocksize: int = serinv_parameters["arrowhead_blocksize"]
            n_diag_blocks: int = serinv_parameters["n_diag_blocks"]

            # Check the model compute parameters
            if diagonal_blocksize is None or n_diag_blocks is None:
                logging.critical("Trying to instanciate Serinv solver on non-ST model.")
                raise ValueError(
                    "Serinv solver is not made for non spatio-temporal models."
                )

            n_processes_solver = self.comm_qeval.size
            if n_processes_solver == 1:
                self.solver = SerinvSolver(
                    config=self.config.solver,
                    diagonal_blocksize=diagonal_blocksize,
                    arrowhead_blocksize=arrowhead_blocksize,
                    n_diag_blocks=n_diag_blocks,
                )
            else:
                # Distributed solver checks
                if not backend_flags["mpi_avail"]:
                    raise ValueError(
                        "Distributed solver is requested but MPI is not available."
                    )
                if n_diag_blocks < n_processes_solver * 3:
                    raise ValueError(
                        f"Not enough diagonal blocks ({n_diag_blocks}) to use the distributed solver with {n_processes_solver} processes."
                    )

                self.nccl_comm = None
                if backend_flags["nccl_avail"]:
                    # --- Initialize NCCL communicator
                    if self.comm_qeval.rank == 0:
                        print(
                            f"rank {self.initial_comm_world.rank} initializing NCCL communicator.",
                            flush=True,
                        )
                        nccl_id = nccl.get_unique_id()
                        self.comm_qeval.bcast(nccl_id, root=0)
                    else:
                        nccl_id = self.comm_qeval.bcast(None, root=0)

                    self.nccl_comm = nccl.NcclCommunicator(
                        self.comm_qeval.size,
                        nccl_id,
                        self.comm_qeval.rank,
                    )
                    synchronize(comm=self.comm_world)

                self.solver = DistSerinvSolver(
                    config=self.config.solver,
                    diagonal_blocksize=diagonal_blocksize,
                    arrowhead_blocksize=arrowhead_blocksize,
                    n_diag_blocks=n_diag_blocks,
                    comm=self.comm_qeval,
                    nccl_comm=self.nccl_comm,
                )

        # --- Set up recurrent variables
        self.gradient_f = xp.zeros(self.model.n_hyperparameters, dtype=xp.float64)
        self.f_values_i = xp.zeros(self.n_f_evaluations, dtype=xp.float64)
        self.eps_mat = xp.zeros(
            (self.model.n_hyperparameters, self.model.n_hyperparameters),
            dtype=xp.float64,
        )
        self.theta_mat = xp.zeros(
            (self.model.theta_internal.size, self.n_f_evaluations), dtype=xp.float64
        )
        self.theta_optimizer = xp.zeros_like(self.model.theta_internal)
        self.theta_optimizer[:] = self.model.theta_internal
        self.theta_star = None # mode not yet computed
        self.theta_star_internal = None
        self.x_star = None # mode not yet computed
        self.cov_theta_internal = None # covariance not yet computed

        # --- Metrics
        self.f_values: ArrayLike = []
        self.theta_values_internal: ArrayLike = []
        self.objective_function_time: ArrayLike = []
        self.solver_time: ArrayLike = []
        self.construction_time: ArrayLike = []
        self.accepted_iter = 0

        # --- Timers
        self.t_construction_qprior = 0.0
        self.t_construction_qconditional = 0.0
        self.solver.t_factorize = 0.0
        self.solver.t_solve = 0.0
        self._print_init()

        logging.info("DALIA initialized.")
        print_msg("DALIA initialized.", flush=True)

    def _print_init(self) -> None:
        """
        Print informations about the DALIA solver.
        """
        str_representation = ""

        # DALIA Header
        str_representation += ascii_logo()

        # Parallelization strategies header
        parallel_strategies_values = [
            [
                "Participating Processes / Total Processes",
                f"{self.world_size} / {self.initial_comm_world.size}",
            ],
            [
                "Parallelization through F()",
                f"{self.world_size // self.comm_feval.size}",
            ],
            [
                "Parallelization through Q()",
                f"{self.comm_feval.size // self.comm_qeval.size}",
            ],
            ["Parallelization through S()", f"{self.comm_qeval.size}"],
        ]
        parallel_strategies_table = tabulate(
            parallel_strategies_values,
            tablefmt="fancy_grid",
            colalign=("left", "center"),
        )
        parallel_strategies_table = add_str_header(
            title="Parallelization strategies",
            table=parallel_strategies_table,
        )
        str_representation += "\n" + boxify(parallel_strategies_table)

        # HPC modules header
        hpc_modules_values = [
            ["Array module", xp.__name__],
            ["MPI available", backend_flags["mpi_avail"]],
            ["Is MPI CUDA aware", backend_flags["mpi_cuda_aware"]],
            ["Is NCCL available", backend_flags["nccl_avail"]],
        ]
        hpc_modules_table = tabulate(
            hpc_modules_values,
            tablefmt="fancy_grid",
            colalign=("left", "center"),
        )
        hpc_modules_table = add_str_header(
            title="Enabled Performance modules",
            table=hpc_modules_table,
        )
        str_representation += "\n" + boxify(hpc_modules_table)

        # Memory usage header
        used_memory, available_memory = memory_report()
        memory_usage_values = [
            ["Solver memory", format_size(self.solver.get_solver_memory())],
            ["Total memory used", format_size(used_memory)],
            ["Total memory available", format_size(available_memory)],
        ]
        memory_usage_table = tabulate(
            memory_usage_values,
            tablefmt="fancy_grid",
            colalign=("left", "center"),
        )
        memory_usage_table = add_str_header(
            title="Memory Report",
            table=memory_usage_table,
        )
        str_representation += "\n" + boxify(memory_usage_table)

        print_msg(str_representation, flush=True)

    def run(self) -> dict:
        """Run the DALIA"""
        synchronize(comm=self.comm_world)
        tic = time.perf_counter()

        # compute mode of the hyperparameters theta
        minimization_result = self.minimize()

        self.theta_star = minimization_result["theta"]
        self.theta_star_internal = minimization_result["theta_internal"]
        self.x_star = minimization_result["x"]

        print("Finished the optimization procedure.")

        # need to update theta_star and x_star to be the same across all ranks
        bcast(data=self.theta_star[:], root=0, comm=self.comm_world)
        bcast(data=self.theta_star_internal[:], root=0, comm=self.comm_world)
        bcast(data=self.x_star[:], root=0, comm=self.comm_world)

        # compute covariance of the hyperparameters theta at the mode
        self.cov_theta_internal = self.compute_covariance_hp(self.theta_star)
        print("Computed covariance of the hyperparameters at the mode.")

        # compute marginal variances of the latent parameters
        marginal_variances_latent = self.get_marginal_variances_latent_parameters(
            self.theta_star, self.x_star
        )
        print_msg("Computed marginal variances of the latent parameters.")

        # compute marginal variances of the observations
        # TODO: only run by default when dense multiplcation issue is fixed, see issue #78
        # marginal_variances_observations = self.get_marginal_variances_observations(
        #     theta_star, x_star
        # )

        # construct new dictionary with the results
        results = {
            "theta": minimization_result["theta"],
            "theta_internal": minimization_result["theta_internal"],
            "x": minimization_result["x"],
            "f": minimization_result["f"],
            "grad_f": minimization_result["grad_f"],
            "f_values": minimization_result["f_values"],
            "theta_values": minimization_result["theta_values"],
            "cov_theta_internal": self.cov_theta_internal,
            "marginal_variances_latent": marginal_variances_latent,
            "optimization_iterations": self.accepted_iter,
            # "marginal_variances_observations": get_host(
            #     marginal_variances_observations
            # ),
        }
        synchronize(comm=self.comm_world)
        toc = time.perf_counter()
        print_msg(f"DALIA inference took: {toc - tic:0.4f} (s)", flush=True)
        return results

    def minimize(self) -> optimize.OptimizeResult:
        """find mode hyperparameters.

        Parameters
        ----------
        None

        Returns
        -------
        minimization_result : scipy.optimize.OptimizeResult
            Result of the optimization procedure.
        """
        # Ensure that all ranks are initialized to the same theta
        check_vector_consistency(
            value=self.model.theta_external,
            comm=self.comm_world,
            flag="self.model.theta_external",
            verbose="Full",
        )

        if len(self.model.theta_external) == 0:
            # Only run the inner iteration
            print_msg("No hyperparameters, just running inner iteration.")
            self.f_value = self._evaluate_f(self.model.theta_external)
            self.minimization_result: dict = {
                "theta_internal": copy.deepcopy(self.model.theta_internal),
                "theta": copy.deepcopy(self.model.theta_external),
                "x": copy.deepcopy(self.model.x),  # [self.model.inverse_permutation_latent_variables],
                "f": copy.deepcopy(self.f_value),
                "grad_f": [],
                "f_values": [],
                ### these values are in internal scale (!!)
                "theta_values": [],
            }

        else:
            print_msg("Starting optimization.")
            self.iter = 0
            self.accepted_iter = 0

            # Define a custom exception to signal early exit
            class OptimizationConvergedEarlyExit(Exception):
                pass

            # Start the minimization procedure
            def callback(intermediate_result: optimize.OptimizeResult):
                theta_i = intermediate_result.x.copy()
                fun_i = intermediate_result.fun
                self.accepted_iter += 1

                # Format the output
                theta_str = ", ".join(f"{theta: .6f}" for theta in theta_i)
                gradient_str = ", ".join(
                    f"{grad: .6f}" for grad in get_host(self.gradient_f)
                )

                print(
                    f"comm_rank: {comm_rank} | "
                    f"Iteration: {self.accepted_iter:2d} (took: {self.objective_function_time[-1]:.2f}) | "
                    f"Theta: [{theta_str}] | "
                    f"Function Value: {fun_i: .6f} | "
                    #f"Gradient: [{gradient_str}] | ",
                    f"Norm(Grad): [{xp.linalg.norm(self.gradient_f): .6f}]",
                    flush=True,
                )

                self.theta_values_internal.append(theta_i)
                self.f_values.append(fun_i)

                # check if f_values have been decreasing over last iterations
                if self.accepted_iter > self.config.f_reduction_lag:
                    if (
                        xp.abs(self.f_values[-self.config.f_reduction_lag] - fun_i)
                        < self.config.f_reduction_tol
                    ):
                        print_msg(
                            f"Optimization converged!  "
                            f"|| f({self.accepted_iter - self.config.f_reduction_lag}) - f({self.accepted_iter}) || = "
                            f"{self.f_values[self.accepted_iter - self.config.f_reduction_lag] - self.f_values[self.accepted_iter-1]:.6f} "
                            f"< {self.config.f_reduction_tol}. Function value: {fun_i:.6f}\n",
                            flush=True,
                        )

                        self.minimization_result = {
                            "theta_internal": copy.deepcopy(self.model.theta_internal),
                            "theta":
                                copy.deepcopy(self.model.theta_external),
                            "x": copy.deepcopy(self.model.x),
                            "f": copy.deepcopy(fun_i),
                            "grad_f": copy.deepcopy(self.gradient_f),
                            "f_values": self.f_values,
                            ### these values are in internal scale (!!)
                            "theta_values": self.theta_values_internal,
                        }

                        raise OptimizationConvergedEarlyExit()

                if self.accepted_iter > self.config.theta_reduction_lag:
                    if (
                        xp.linalg.norm(
                            self.theta_values_internal[-self.config.theta_reduction_lag]
                            - theta_i
                        )
                        < self.config.theta_reduction_tol
                    ):
                        norm_diff = xp.linalg.norm(
                            self.theta_values_internal[
                                self.accepted_iter - self.config.theta_reduction_lag
                            ]
                            - theta_i
                        )
                        print_msg(
                            f"Optimization converged!  "
                            f"|| theta({self.accepted_iter - self.config.theta_reduction_lag}) - theta({self.accepted_iter}) || = "
                            f"{norm_diff:.6f} "
                            f"< {self.config.theta_reduction_tol}. Function value: {fun_i:.6f}\n",
                            flush=True,
                        )

                        self.minimization_result = {
                            "theta_internal": get_host(self.model.theta_internal),
                            "theta": get_host(
                                self.model.theta_external),
                            "x": get_host(
                                self.model.x
                                # self.model.x[
                                #     self.model.inverse_permutation_latent_variables
                                # ]
                            ),
                            "f": copy.deepcopy(fun_i),
                            "grad_f": copy.deepcopy(self.gradient_f),
                            "f_values": copy.deepcopy(self.f_values),
                            "theta_values": copy.deepcopy(self.theta_values_internal),
                        }

                        raise OptimizationConvergedEarlyExit()

            try:
                scipy_result = optimize.minimize(
                    fun=self._objective_function,
                    x0=get_host(self.theta_optimizer),
                    method="L-BFGS-B",
                    jac=self.config.minimize.jac,
                    options={
                        "maxiter": self.config.minimize.max_iter,
                        "maxcor": self.config.minimize.maxcor,
                        "maxls": self.config.minimize.maxls,
                        "gtol": self.config.minimize.gtol,
                        "disp": self.config.minimize.disp,
                        "ftol": 1e-22,
                    },
                    callback=callback,
                )
            except OptimizationConvergedEarlyExit:
                return self.minimization_result

            print(
                f"rank {comm_rank} | objective function time: {self.objective_function_time[1:]}"
            )

            # MEMO:
            # From here rank 0 own the optimized theta_star and the
            # corresponding x_star. Other ranks own garbage thetas in
            # their self.model.theta
            if scipy_result.success:
                print_msg(
                    "Optimization converged successfully after",
                    self.iter,
                    "iterations.\n",
                    "SUCCESS MSG: ",
                    scipy_result.message,
                    flush=True,
                )
            else:
                print_msg(
                    "Optimization did not converge.",
                    "FAILURE MSG: ",
                    scipy_result.message,
                    flush=True,
                )

            self.minimization_result: dict = {
                "theta_internal": copy.deepcopy(self.model.theta_internal), #  scipy_result.x, #
                "theta": copy.deepcopy(self.model.theta_external),
                "x": copy.deepcopy(self.model.x),  # [self.model.inverse_permutation_latent_variables]
                "f": copy.deepcopy(scipy_result.fun),
                "grad_f": copy.deepcopy(self.gradient_f),
                "f_values": copy.deepcopy(self.f_values),
                "theta_values": copy.deepcopy(self.theta_values_internal),
            }

        return self.minimization_result

    def _objective_function(
        self,
        theta_i: NDArray,
    ) -> tuple:
        """Objective function to optimize.

        Parameters
        ----------
        theta_i : NDArray
            Hyperparameters theta.

        Returns
        -------
        objective_function_evalutation : tuple
            Function value f(theta) evaluated at theta_i and its gradient.
        """

        self.t_construction_qprior = 0.0
        self.t_construction_qconditional = 0.0
        self.solver.t_factorize = 0.0
        self.solver.t_solve = 0.0

        synchronize(comm=self.comm_world)
        tic = time.perf_counter()
        # Generate theta matrix with different theta's to evaluate
        # currently central difference scheme is used for gradient
        self.f_values_i[:] = 0.0

        # Multiprocessing task to rank assignment
        n_feval_comm = self.world_size // self.comm_feval.size
        task_mapping = []
        for i in range(self.n_f_evaluations):
            task_mapping.append(i % n_feval_comm)

        # Initialize central difference scheme matrix
        self.eps_mat[:] = self.eps_gradient_f * xp.eye(self.model.n_hyperparameters)
        self.theta_mat[:] = xp.repeat(
            get_device(theta_i).reshape(-1, 1), self.n_f_evaluations, axis=1
        )
        self.theta_mat[:, 1 : 1 + self.model.n_hyperparameters] += self.eps_mat
        self.theta_mat[
            :, self.model.n_hyperparameters + 1 : self.n_f_evaluations
        ] -= self.eps_mat

        # Proceed to the parallel function evaluation
        for feval_i in range(self.n_f_evaluations - 1, -1, -1):
            # Perform the evaluation in reverse order so that the stored and returned
            # self.x value matches the "bare" hyperparameters evaluation
            if self.color_feval == task_mapping[feval_i]:
                self.f_values_i[feval_i] = self._evaluate_f(
                    theta_i=self.theta_mat[:, feval_i]
                )

        # Here carefull on the reduction as it's gonna add the values from all ranks and not only the root of the groups - TODO
        synchronize(comm=self.comm_world)
        allreduce(
            self.f_values_i,
            op="sum",
            factor=1 / self.comm_feval.size,
            comm=self.comm_world,
        )
        synchronize(comm=self.comm_world)

        # Compute gradient using central difference scheme
        for i in range(self.model.n_hyperparameters):
            self.gradient_f[i] = (
                self.f_values_i[i + 1]
                - self.f_values_i[self.model.n_hyperparameters + i + 1]
            ) / (2 * self.eps_gradient_f)

        f_0 = get_host(self.f_values_i[0])
        grad_f = get_host(self.gradient_f)

        synchronize(comm=self.comm_world)
        toc = time.perf_counter()
        self.objective_function_time.append(toc - tic)
        self.solver_time.append(self.solver.t_factorize + self.solver.t_solve)
        self.construction_time.append(
            self.t_construction_qprior + self.t_construction_qconditional
        )

        if self.iter > 0 and self.verbosity > 0:
            print(
                f"rank {comm_rank} | objfunc_time: {self.objective_function_time[1:]} | solver_time: {self.solver_time[1:]} | construction_time: {self.construction_time[1:]}",
                flush=True,
            )
        self.iter += 1

        return (f_0, grad_f)

    def _evaluate_f(
        self,
        theta_i: NDArray,
    ) -> float:
        """Evaluate the objective function f(theta) = log(p(theta|y)).

        Parameters
        ----------
        theta_i : NDArray
            Hyperparameters theta.

        Returns
        -------
        objective_function_evalutation : float
            Function value f(theta) evaluated at theta_i.

        Notes
        -----
        The objective function f(theta) is an approximation of the
        log posterior of the hyperparameters theta evaluated at theta_i
        in log-scale. Consisting of the following 4 terms: log prior
        hyperparameters, log likelihood, log prior of the latent parameters,
        and log conditional of the latent parameters.
        """
        import time

        tic = time.time()

        # self.model.theta_internal[:] = theta_i
        self.model.theta_internal = theta_i
        f_theta = xp.zeros(1, dtype=xp.float64)

        # --- Optimize x and evaluate the conditional of the latent parameters
        if self.model.is_likelihood_gaussian():
            # Done by both processes
            synchronize_gpu()
            tic = time.perf_counter()
            self.model.construct_Q_prior()
            synchronize_gpu()
            toc = time.perf_counter()
            self.t_construction_qprior += toc - tic

            eta = xp.zeros_like(self.model.y, dtype=xp.float64)
            x = xp.zeros_like(self.model.x, dtype=xp.float64)

            n_qeval_comm = self.qeval_world.size // self.comm_qeval.size
            task_mapping = [i % n_qeval_comm for i in range(2)]

            if task_mapping[0] == self.color_qeval:

                # Done by processes "even"
                synchronize_gpu()
                tic = time.perf_counter()
                Q_conditional = self.model.construct_Q_conditional(eta)
                synchronize_gpu()
                toc = time.perf_counter()
                self.t_construction_qconditional += toc - tic

                self.solver.factorize(A=Q_conditional, sparsity="bta")

                rhs: NDArray = self.model.construct_information_vector(
                    eta,
                    x,
                )

                self.model.x[:] = self.solver.solve(
                    rhs=rhs,
                    sparsity="bta",
                )

                conditional_latent_parameters = (
                    self._evaluate_conditional_latent_parameters(
                        Q_conditional=Q_conditional,
                        x=None,
                        x_mean=self.model.x,
                    )
                )

                f_theta[0] += conditional_latent_parameters
            if task_mapping[1] == self.color_qeval:
                # Done by processes "odd"
                log_prior_hyperparameters: float = (
                    self.model.evaluate_log_prior_hyperparameters()
                )

                likelihood: float = self.model.evaluate_likelihood(eta=eta)
                prior_latent_parameters: float = (
                    self._evaluate_prior_latent_parameters()
                )

                f_theta[0] -= (
                    log_prior_hyperparameters + likelihood + prior_latent_parameters
                )

            if task_mapping[0] != task_mapping[1]:
                synchronize(comm=self.comm_qeval)
                allreduce(
                    f_theta,
                    op="sum",
                    factor=1 / self.comm_qeval.size,
                    comm=self.comm_feval,
                )
                synchronize(comm=self.comm_qeval)

        else:
            synchronize_gpu()
            tic = time.perf_counter()
            self.model.construct_Q_prior()
            synchronize_gpu()
            toc = time.perf_counter()
            self.t_construction_qprior += toc - tic

            log_prior_hyperparameters: float = (
                self.model.evaluate_log_prior_hyperparameters()
            )

            Q_conditional, self.model.x[:], eta = self._inner_iteration()

            conditional_latent_parameters = (
                self._evaluate_conditional_latent_parameters(
                    Q_conditional=Q_conditional,
                    x=None,
                    x_mean=None,
                )
            )

            prior_latent_parameters: float = self._evaluate_prior_latent_parameters(
                x=self.model.x,
            )

            likelihood: float = self.model.evaluate_likelihood(
                eta=eta,
            )

            f_theta[0] -= (
                log_prior_hyperparameters
                + likelihood
                + prior_latent_parameters
                - conditional_latent_parameters
            )

        if xp.isnan(f_theta[0]):
            raise ValueError(
                f"Rank: {comm_rank} (theta) is NaN. Check what is happening."
            )

        synchronize(comm=self.comm_feval)
        allreduce(
            f_theta,
            op="sum",
            factor=1 / self.comm_feval.size,
            comm=self.comm_feval,
        )
        synchronize(comm=self.comm_feval)

        return f_theta[0]

    def compute_covariance_hp(self, theta_external: NDArray) -> NDArray:
        """compute the covariance matrix of the hyperparameters theta.

        Parameters
        ----------
        theta_i : NDArray
            Hyperparameters theta.

        Returns
        -------
        cov_theta : NDArray[dim_theta, dim_theta]
            Covariance matrix of the hyperparameters theta.
        """

        # self.model.rescale_hyperparameters_to_internal(theta_interpret, direction="forward")
        # ensure that all ranks are initialized to the same theta
        check_vector_consistency(
            theta_external,
            comm=self.comm_world,
            flag="theta_external",
            verbose="Full",
        )
        print_msg(
            f"Computing covariance of hyperparameters at theta_external {theta_external}.",
            flush=True,
        )

        synchronize(comm=self.comm_world)
        tic = time.perf_counter()
        self.model.theta_external = theta_external

        hess_theta_internal = self._evaluate_hessian_f(self.model.theta_internal)
        print_msg(
            f"hessian_f: \n {hess_theta_internal}",
            flush=True,
        )
        cov_theta_internal = xp.linalg.inv(hess_theta_internal)

        synchronize(comm=self.comm_world)
        toc = time.perf_counter()
        print_msg(
            "Time to compute covariance of hyperparameters:",
            toc - tic,
            flush=True,
        )

        return cov_theta_internal

    def _evaluate_hessian_f(
        self,
        theta_internal: NDArray,
    ) -> NDArray:
        """Approximate the hessian of the function f(theta) = log(p(theta|y)).

        Parameters
        ----------
        theta_i : NDArray
            Hyperparameters theta.

        Returns
        -------
        hessian_f : NDArray[dim_theta, dim_theta]

        Notes
        -----
        Compute finite difference approximation of the hessian of f at theta_i.
        """

        ## TODO: this is the quick fix ...
        # self.model.theta[:] = theta_i
        dim_theta = self.model.n_hyperparameters

        # pre-allocate storage for the hessian & f_values
        hess = xp.zeros((dim_theta, dim_theta), dtype=xp.float64)
        # pre-allocate perturbation matrix
        eps_mat = xp.eye(dim_theta, dtype=xp.float64)
        # TODO: should be we have separate eps_hessian_f?
        eps_mat *= self.eps_hessian_f

        loop_dim = dim_theta * dim_theta

        # store: theta+eps_i, theta, theta-eps_i
        f_ii_loc = xp.zeros((3, dim_theta), dtype=xp.float64)
        # store: theta+eps_i+eps_j, theta+eps_i-eps_j, theta-eps_i+eps_j, theta-eps_i-eps_j
        f_ij_loc = xp.zeros((4, loop_dim), dtype=xp.float64)

        # compute number of necessary function evaluations
        # f(theta), 2*dim_theta for the diagonal, 4*dim_theta*(dim_theta-1)/2 for the off-diagonal
        no_eval = 1 + 2 * dim_theta + 4 * dim_theta * (dim_theta - 1) // 2
        n_feval_comm = self.world_size // self.comm_feval.size
        if n_feval_comm > no_eval:
            print("No idea what happens with MPI split here.")
            raise ValueError("no_eval > 2*loop_dim")

        task_mapping = []
        for i in range(no_eval):
            task_mapping.append(i % n_feval_comm)

        counter = 0
        # compute f(theta)
        if self.color_feval == task_mapping[0]:
            theta_i = theta_internal.copy()
            f_theta = self._evaluate_f(theta_i)
            f_ii_loc[1, :] = f_theta

        counter += 1

        for k in range(loop_dim):
            i = k // dim_theta
            j = k % dim_theta

            # diagonal elements
            if i == j:
                if self.color_feval == task_mapping[counter]:
                    # theta+eps_i
                    # theta_i = theta_internal.copy()
                    # f_ii_loc[0, i] = self._evaluate_f(theta_i + eps_mat[i, :])
                    theta_i = theta_internal + eps_mat[i, :]
                    result = self._evaluate_f(theta_i)
                    f_ii_loc[0, i] = result
                counter += 1

                if self.color_feval == task_mapping[counter]:
                    # theta-eps_i
                    # theta_i = theta_internal.copy()
                    # f_ii_loc[2, i] = self._evaluate_f(theta_i - eps_mat[i, :])
                    theta_i = theta_internal - eps_mat[i, :]
                    result = self._evaluate_f(theta_i)
                    f_ii_loc[2, i] = result
                counter += 1

            # as hessian is symmetric we only have to compute the upper triangle
            elif i < j:
                # theta+eps_i+eps_j
                if self.color_feval == task_mapping[counter]:
                    # theta_i = theta_internal.copy()
                    # f_ij_loc[0, k] = self._evaluate_f(
                    #     theta_i + eps_mat[i, :] + eps_mat[j, :]
                    # )
                    theta_i = theta_internal + eps_mat[i, :] + eps_mat[j, :]
                    f_ij_loc[0, k] = self._evaluate_f(theta_i)
                counter += 1

                # theta+eps_i-eps_j
                if self.color_feval == task_mapping[counter]:
                    # theta_i = theta_internal.copy()
                    # f_ij_loc[1, k] = self._evaluate_f(
                    #     theta_i + eps_mat[i, :] - eps_mat[j, :]
                    # )
                    theta_i = theta_internal + eps_mat[i, :] - eps_mat[j, :]
                    f_ij_loc[1, k] = self._evaluate_f(theta_i)
                counter += 1

                # theta-eps_i+eps_j
                if self.color_feval == task_mapping[counter]:
                    # theta_i = theta_internal.copy()
                    # f_ij_loc[2, k] = self._evaluate_f(
                    #     theta_i - eps_mat[i, :] + eps_mat[j, :]
                    # )
                    theta_i = theta_internal - eps_mat[i, :] + eps_mat[j, :]
                    f_ij_loc[2, k] = self._evaluate_f(theta_i)
                counter += 1

                # theta-eps_i-eps_j
                if self.color_feval == task_mapping[counter]:
                    # theta_i = theta_internal.copy()
                    # f_ij_loc[3, k] = self._evaluate_f(
                    #     theta_i - eps_mat[i, :] - eps_mat[j, :]
                    # )
                    theta_i = theta_internal - eps_mat[i, :] - eps_mat[j, :]
                    f_ij_loc[3, k] = self._evaluate_f(theta_i)
                counter += 1

        allreduce(
            f_ii_loc,
            op="sum",
            comm=self.comm_world,
            factor=1 / self.comm_feval.size,
        )
        allreduce(
            f_ij_loc,
            op="sum",
            comm=self.comm_world,
            factor=1 / self.comm_feval.size,
        )
        synchronize(comm=self.comm_feval)

        # compute hessian
        for k in range(loop_dim):
            i = k // dim_theta
            j = k % dim_theta

            # diagonal elements
            if i == j:
                hess[i, i] = (
                    f_ii_loc[0, i] - 2 * f_ii_loc[1, i] + f_ii_loc[2, i]
                ) / eps_mat[i, i] ** 2
            # as hessian is symmetric we only have to compute the upper triangle
            elif i < j:
                hess[i, j] = (
                    f_ij_loc[0, k] - f_ij_loc[1, k] - f_ij_loc[2, k] + f_ij_loc[3, k]
                ) / (4 * eps_mat[i, i] * eps_mat[j, j])
                hess[j, i] = hess[i, j]

        # compute eigenvalues
        eigvals = xp.linalg.eigvalsh(hess)

        if xp.any(eigvals < 0):
            print_msg(f"Negative eigenvalues detected: {eigvals}")

        return hess

    def marginal_distributions_hp(self, 
                                  #quantiles: NDArray = xp.array([0.0001, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.9999])
                                  quantiles: NDArray = xp.array([0.025, 0.25, 0.5, 0.75, 0.975])
                                  ) -> dict:
        """Compute the marginal distributions of the hyperparameters theta.

        Parameters
        ----------
        quantiles : NDArray
            Quantiles to compute. If not provided, default quantiles are used. If None, no quantiles are computed.

        Returns
        -------
        dict
            Dictionary containing the marginal distributions of the hyperparameters theta and possibly quantiles / percentiles.

        """

        # check that theta_star and covariance matrix are computed
        if self.theta_star is None or self.cov_theta_internal is None or self.x_star is None:
            raise ValueError("theta_star, x_star and covariance matrix of the hyperparameters must be computed before calling marginal_distributions_hp(). Please run the full DALIA pipeline or set them manually.")

        # set up dictionary to store results
        results = {
            'hyperparameters': {},
            'summary': {
                'n_params': self.model.n_hyperparameters,
                'param_names': self.model.theta_keys,
                'quantile_levels': quantiles.tolist() if quantiles is not None else None
            }
        }

        # Import necessary functions
        from dalia.utils.gaussian_quadrature import compute_variance_gauss_hermite
        from dalia.utils.reparametrizations import compute_bounds, compute_transformed_pdf, compute_transformed_quantiles
        from dalia.prior_hyperparameters import GaussianMVNPriorHyperparameters

        hp_offset = 0
        for i, prior in enumerate(self.model.prior_hyperparameters):
            if isinstance(prior, GaussianMVNPriorHyperparameters):
                n_hp_for_this_prior = prior.mean.shape[0]
            else:
                n_hp_for_this_prior = 1

            for j in range(0, n_hp_for_this_prior, 1):
                param_name = results['summary']['param_names'][i+hp_offset+j]

                # Extract marginal parameters for this hyperparameter
                theta_internal_i = self.theta_star_internal[i+hp_offset+j]
                marg_var_internal_i = self.cov_theta_internal[
                    i + hp_offset + j, i + hp_offset + j
                ]

                # compute external_mean and external_var using
                # compute_variance_gauss_hermite(mean_internal, variance_internal, transform, n_points=20): from utils gaussian quadrature
                gauss_hermite_result = compute_variance_gauss_hermite(
                    theta_internal_i, marg_var_internal_i, prior.rescale_hyperparameters_to_internal, n_points=30
                )

                # compute bounds for theta intervals using compute_bounds() from utils
                (theta_internal_lower, theta_internal_upper), (theta_external_lower, theta_external_upper) = compute_bounds(
                    theta_internal_i, marg_var_internal_i, prior.rescale_hyperparameters_to_internal, n_std=4
                )

                # set theta_internal_interval
                theta_internal_interval = xp.linspace(theta_internal_lower, theta_internal_upper, num=100)

                # Compute PDF values in external scale
                theta_external_interval, pdf_external = compute_transformed_pdf(theta_internal_i, marg_var_internal_i, theta_internal_interval, prior.rescale_hyperparameters_to_internal)

                # Initialize parameter dictionary
                param_dict = {
                    'mean_internal': float(get_host(theta_internal_i)),
                    'variance_internal': float(get_host(marg_var_internal_i)),
                    'mean_external': float(get_host(gauss_hermite_result['mean'])),
                    'variance_external': float(get_host(gauss_hermite_result['variance'])),
                    'pdf_data': (get_host(theta_external_interval), get_host(pdf_external))  # tuple of xp arrays
                }

                # if quantiles is not None, compute quantiles using compute_transformed_quantiles()
                if quantiles is not None:
                    quantiles_external = compute_transformed_quantiles(
                        theta_internal_i, marg_var_internal_i, quantiles, prior.rescale_hyperparameters_to_internal
                    )

                    # Also compute internal quantiles for completeness
                    from scipy.stats import norm
                    quantiles_internal = get_device(norm.ppf(get_host(quantiles), loc=get_host(theta_internal_i), scale=get_host(xp.sqrt(marg_var_internal_i))))

                    param_dict['quantiles'] = {
                        'levels': get_host(quantiles).tolist(),
                        'internal': {
                            'values': get_host(quantiles_internal).tolist(),
                            'pairs': list(zip(get_host(quantiles).tolist(), get_host(quantiles_internal).tolist()))
                        },
                        'external': {
                            'values': get_host(quantiles_external).tolist(),
                            'pairs': list(zip(get_host(quantiles).tolist(), get_host(quantiles_external).tolist()))
                        }
                    }

                # Store in main results dictionary
                results['hyperparameters'][param_name] = param_dict

            hp_offset += n_hp_for_this_prior-1

        # Old code
        if False:
            # iterate over all hyperparameters and store outputs in a dictionary
            for i in range(self.model.n_hyperparameters):
                param_name = results['summary']['param_names'][i]

                # Extract marginal parameters for this hyperparameter
                theta_internal_i = self.theta_star_internal[i]
                marg_var_internal_i = self.cov_theta_internal[i, i]

                # compute external_mean and external_var using
                # compute_variance_gauss_hermite(mean_internal, variance_internal, transform, n_points=20): from utils gaussian quadrature
                gauss_hermite_result = compute_variance_gauss_hermite(
                    theta_internal_i, marg_var_internal_i, self.model.prior_hyperparameters[i].rescale_hyperparameters_to_internal, n_points=30
                )

                # compute bounds for theta intervals using compute_bounds() from utils
                (theta_internal_lower, theta_internal_upper), (theta_external_lower, theta_external_upper) = compute_bounds(
                    theta_internal_i, marg_var_internal_i, self.model.prior_hyperparameters[i].rescale_hyperparameters_to_internal, n_std=4
                )

                # set theta_internal_interval
                theta_internal_interval = xp.linspace(theta_internal_lower, theta_internal_upper, num=100)

                # Compute PDF values in external scale
                theta_external_interval, pdf_external = compute_transformed_pdf(theta_internal_i, marg_var_internal_i, theta_internal_interval, self.model.prior_hyperparameters[i].rescale_hyperparameters_to_internal)

                # Initialize parameter dictionary
                param_dict = {
                    'mean_internal': float(get_host(theta_internal_i)),
                    'variance_internal': float(get_host(marg_var_internal_i)),
                    'mean_external': float(get_host(gauss_hermite_result['mean'])),
                    'variance_external': float(get_host(gauss_hermite_result['variance'])),
                    'pdf_data': (get_host(theta_external_interval), get_host(pdf_external))  # tuple of xp arrays
                }

                # if quantiles is not None, compute quantiles using compute_transformed_quantiles()
                if quantiles is not None:
                    quantiles_external = compute_transformed_quantiles(
                        theta_internal_i, marg_var_internal_i, quantiles, self.model.prior_hyperparameters[i].rescale_hyperparameters_to_internal
                    )

                    # Also compute internal quantiles for completeness
                    from scipy.stats import norm
                    quantiles_internal = get_device(norm.ppf(get_host(quantiles), loc=get_host(theta_internal_i), scale=get_host(xp.sqrt(marg_var_internal_i))))

                    param_dict['quantiles'] = {
                        'levels': get_host(quantiles).tolist(),
                        'internal': {
                            'values': get_host(quantiles_internal).tolist(),
                            'pairs': list(zip(get_host(quantiles).tolist(), get_host(quantiles_internal).tolist()))
                        },
                        'external': {
                            'values': get_host(quantiles_external).tolist(),
                            'pairs': list(zip(get_host(quantiles).tolist(), get_host(quantiles_external).tolist()))
                        }
                    }

                # Store in main results dictionary
                results['hyperparameters'][param_name] = param_dict

        # return dictionary
        return results

    def _compute_covariance_latent_parameters(
        self, theta_internal: NDArray, x_star: NDArray
    ) -> None:
        """Compute the marginal distribution of the latent parameters x.

        Parameters
        ----------
        theta_i : NDArray
            Hyperparameters theta.
        x_star : NDArray
            Latent parameters x(theta_i).

        Returns
        -------
        marginal_latent_parameters : NDArray
            Marginal distribution of the latent parameters x.
        """

        self.model.theta_internal = xp.atleast_1d(theta_internal)
        self.model.x[:] = x_star

        eta = self.model.a @ self.model.x

        synchronize_gpu()
        tic = time.perf_counter()
        self.model.construct_Q_conditional(eta)
        synchronize_gpu()
        toc = time.perf_counter()
        self.t_construction_qconditional += toc - tic

        self.solver.factorize(self.model.Q_conditional, sparsity="bta")
        self.solver.selected_inversion(sparsity="bta")

    def get_marginal_variances_latent_parameters(
        self, theta_external: NDArray = None, x_star: NDArray = None
    ) -> NDArray:

        # TODO: this should be only called by rank 0?
        if theta_external is None and x_star is None:
            print(
                "Computing marginal variances for currently stored latent parameters. "
            )
            x_star = self.model.x
            theta = self.model.theta_internal
        elif theta_external is not None and x_star is not None:
            ## assume theta to be in "external" scale
            self.model.theta_external = xp.atleast_1d(theta_external)
            theta = self.model.theta_internal

        elif theta is None or x_star is None:
            raise ValueError(
                "BOTH or NEITHER theta and x_star must be provided to compute the marginal variances."
            )

        check_vector_consistency(theta, comm=self.comm_world, flag="theta", verbose="Full")
        check_vector_consistency(x_star, comm=self.comm_world, flag="x_star", verbose="Minimal")

        # check order x_star ... -> potentially need to reorder marginal variances
        self._compute_covariance_latent_parameters(theta, x_star)

        # now only extract diagonal elements corresponding to marginal variances of the latent parameters
        marginal_variances_sp = self.solver._structured_to_spmatrix(
            sp.sparse.eye(self.model.n_latent_parameters, dtype=xp.float64),
            sparsity="bta",
        )

        marginal_variances = extract_diagonal(marginal_variances_sp)

        return marginal_variances

    def get_marginal_variances_observations(
        self, theta_external: NDArray = None, x_star: NDArray = None
    ) -> NDArray:
        """Extract the marginal variances of the observations.

        Parameters
        ----------
        theta_i : NDArray
            Hyperparameters theta.
        x_star : NDArray
            Latent parameters x(theta_i).

        Notes
        -----

        Cov(y) = Cov(Ax) = A Cov(x) A^T = A Q_selected_inv A^T
        -> diag(Cov(y)) = diag(A Q_selected_inv A^T)

        Returns
        -------
        marginal_variances_observations : NDArray
            Marginal variances of the observations.
        """

        # TODO: implement this for non-Gaussian likelihoods
        check_vector_consistency(theta_external, comm=self.comm_world, flag="theta_external", verbose="Full")
        check_vector_consistency(x_star, comm=self.comm_world, flag="x_star", verbose="Minimal")

        if self.model.is_likelihood_gaussian():
            # TODO: this should be only called by rank 0?
            if theta_external is None and x_star is None:
                print(
                    "Computing marginal variances for currently stored latent parameters. "
                )
                x_star = self.model.x
                theta_external = self.model.theta_external

            if theta_external is None or x_star is None:
                raise ValueError(
                    "BOTH or NEITHER theta and x_star must be provided to compute the marginal variances."
                )

                # check order x_star ... -> potentially need to reorder marginal variances
            self._compute_covariance_latent_parameters(theta_external, x_star)

            # now only extract diagonal elements corresponding to marginal variances of the latent parameters
            variances_latent = self.solver._structured_to_spmatrix(
                self.model.Q_conditional,
                sparsity="bta",
            )

            # compute diag(A Q_selected_inv A^T)
            # TODO: sparsify this. can be improved A LOT
            marginal_variances_observations = (
                self.model.a @ variances_latent @ self.model.a.T
            ).diagonal()

            return marginal_variances_observations

        raise NotImplementedError(
            "in compute marginals observations: Only Gaussian likelihood is currently supported."
        )

    def _inner_iteration(
        self,
    ) -> float:
        """Inner iteration to optimize the latent parameters x.

        Parameters
        ----------
        None

        Returns
        -------
        logdet : float
            Log determinant of the conditional precision matrix Q_conditional.
        """
        x_star = self.model.x.copy()
        x_update = xp.zeros_like(self.model.x, dtype=xp.float64)
        x_i_norm: float = 1.0
        eta = xp.zeros_like(self.model.y, dtype=xp.float64)

        counter: int = 0
        while x_i_norm >= self.eps_inner_iteration:
            if counter > self.inner_iteration_max_iter:
                print_msg(
                    "Theta value at failing of the inner_iteration: ",
                    self.model.theta_internal,
                    flush=True,
                )
                raise ValueError(
                    f"Inner iteration did not converge after {counter} iterations."
                )

            x_star[:] += x_update
            eta[:] = self.model.a @ x_star

            synchronize_gpu()
            tic = time.perf_counter()
            Q_conditional = self.model.construct_Q_conditional(eta)
            synchronize_gpu()
            toc = time.perf_counter()
            self.t_construction_qconditional += toc - tic

            self.solver.factorize(A=Q_conditional, sparsity="bta")

            rhs: NDArray = self.model.construct_information_vector(
                eta,
                x_star,
            )
            x_update[:] = self.solver.solve(
                rhs=rhs,
                sparsity="bta",
            )

            x_i_norm = xp.linalg.norm(x_update)
            counter += 1

        return Q_conditional, x_star, eta

    def _evaluate_prior_latent_parameters(
        self,
        x: NDArray = None,
    ) -> float:
        """Evaluation of the prior of the latent parameters at x using
        the prior precision matrix Q_prior and assuming mean zero.

        Parameters
        ----------
        x : NDArray
            Latent parameters.

        Returns
        -------
        logprior : float
            Log prior of the latent parameters evaluated at x

        Notes
        -----
        The prior of the latent parameters is by definition a multivariate normal
        distribution with mean 0 and precision matrix Q_prior which is evaluated at
        x in log-scale. The evaluation requires the computation of the log
        determinant of Q_prior.
        Log normal:
        .. math:: 0.5*log(1/(2*pi)^n * |Q_prior|)) - 0.5 * x.T Q_prior x
        """
        self.solver.factorize(self.model.Q_prior, sparsity="bt")
        logdet_Q_prior: float = self.solver.logdet(sparsity="bt")

        log_prior_latent_parameters: float = +0.5 * logdet_Q_prior

        if x is not None:
            log_prior_latent_parameters -= 0.5 * x.T @ self.model.Q_prior @ x

        return log_prior_latent_parameters

    def _evaluate_conditional_latent_parameters(
        self,
        Q_conditional: NDArray,
        x: NDArray = None,
        x_mean: NDArray = None,
    ) -> float:
        """Evaluation of the conditional of the latent parameters at x using
        the conditional precision matrix Q_conditional and the mean x_mean.

        Parameters
        ----------
        Q_conditional : NDArray
            Conditional precision matrix.
        x : NDArray
            Latent parameters.
        x_mean : NDArray
            Mean of the latent parameters.

        Returns
        -------
        log_conditional : float
            Log conditional of the latent parameters evaluated at x

        Notes
        -----
        The conditional of the latent parameters is by definition a multivariate normal distribution with mean
        x_mean and precision matrix Q_conditional which is evaluated at x in log-scale.
        The evaluation requires the computation of the log determinant of Q_conditional.
        log normal: 0.5*log(1/(2*pi)^n * |Q_conditional|)) - 0.5 * (x - x_mean).T @ Q_conditional @ (x - x_mean)
        """
        # Compute the log determinant of Q_conditional
        logdet_Q_conditional = self.solver.logdet(sparsity="bta")

        if x is None and x_mean is None:
            quadratic_form = 0.0
        # TODO: there is probably a cleaner way to formulate these statements ...
        # the else fails if x_mean is None
        else:
            # Symmetrizing (averaging the tip of the arrow to tame down numerical innaccuracies)
            tip_accu = x_mean[-self.model.total_number_fixed_effects() :].copy()
            synchronize(comm=self.comm_qeval)
            allreduce(
                tip_accu,
                op="sum",
                factor=1 / self.comm_qeval.size,
                comm=self.comm_qeval,
            )
            synchronize(comm=self.comm_qeval)
            x_mean[-self.model.total_number_fixed_effects() :] = tip_accu

            if x is None and x_mean is not None:
                quadratic_form = x_mean.T @ Q_conditional @ x_mean
            else:
                quadratic_form = (x - x_mean).T @ Q_conditional @ (x - x_mean)

        # Compute the log conditional
        log_conditional = 0.5 * logdet_Q_conditional - 0.5 * quadratic_form

        return log_conditional
