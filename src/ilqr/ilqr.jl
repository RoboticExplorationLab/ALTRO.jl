
"""
    iLQRSolver

A fast solver for unconstrained trajectory optimization that uses a Riccati recursion
to solve for a local feedback controller around the current trajectory, and then 
simulates the system forward using the derived feedback control law.

# Constructor
    Altro.iLQRSolver(prob, opts; kwarg_opts...)
"""
struct iLQRSolver{T,I<:QuadratureRule,L,O,n,n̄,m,L1,C} <: UnconstrainedSolver{T}
    # Model + Objective
    model::L
    obj::O

    # Problem info
    x0::MVector{n,T}
    xf::MVector{n,T}
    tf::T
    N::Int

    opts::SolverOptions{T}
    stats::SolverStats{T}

    # Primal Duals
    Z::Traj{n,m,T,KnotPoint{T,n,m,L1}}
    Z̄::Traj{n,m,T,KnotPoint{T,n,m,L1}}

    # Data variables
    # K::Vector{SMatrix{m,n̄,T,L2}}  # State feedback gains (m,n,N-1)
    K::Vector{SizedMatrix{m,n̄,T,2,Matrix{T}}}  # State feedback gains (m,n,N-1)
    d::Vector{SizedVector{m,T,Vector{T}}}  # Feedforward gains (m,N-1)

    D::Vector{DynamicsExpansion{T,n,n̄,m}}  # discrete dynamics jacobian (block) (n,n+m+1,N)
    G::Vector{SizedMatrix{n,n̄,T,2,Matrix{T}}}        # state difference jacobian (n̄, n)

	quad_obj::TO.CostExpansion{n,m,T}  # quadratic expansion of obj
	S::TO.CostExpansion{n̄,m,T}         # Cost-to-go expansion
    E::TO.CostExpansion{n̄,m,T}         # cost expansion 
    Q::TO.CostExpansion{n̄,m,T}         # Action-value expansion
    Qprev::TO.CostExpansion{n̄,m,T}     # Action-value expansion from previous iteration

    # Q_tmp::TO.QuadraticCost{n̄,m,T,SizedMatrix{n̄,n̄,T,2,Matrix{T}},SizedMatrix{m,m,T,2,Matrix{T}}}
    Q_tmp::TO.Expansion{n̄,m,T}
	Quu_reg::SizedMatrix{m,m,T,2,Matrix{T}}
	Qux_reg::SizedMatrix{m,n̄,T,2,Matrix{T}}
    ρ::Vector{T}   # Regularization
    dρ::Vector{T}  # Regularization rate of change

    cache::FiniteDiff.JacobianCache{Vector{T}, Vector{T}, Vector{T}, UnitRange{Int}, Nothing, Val{:forward}(), T}
    exp_cache::C
    grad::Vector{T}  # Gradient

    logger::SolverLogger
end

function iLQRSolver(
        prob::Problem{QUAD,T}, 
        opts::SolverOptions=SolverOptions(), 
        stats::SolverStats=SolverStats(parent=solvername(iLQRSolver));
        kwarg_opts...
    ) where {QUAD,T}
    set_options!(opts; kwarg_opts...)

    # Init solver results
    n,m,N = size(prob)
    n̄ = RobotDynamics.state_diff_size(prob.model)

    x0 = prob.x0
    xf = prob.xf

    Z = prob.Z
    # Z̄ = Traj(n,m,Z[1].dt,N)
    Z̄ = copy(prob.Z)

	K = [zeros(T,m,n̄) for k = 1:N-1]
    d = [zeros(T,m)   for k = 1:N-1]

	D = [DynamicsExpansion{T}(n,n̄,m) for k = 1:N-1]
	G = [SizedMatrix{n,n̄}(zeros(n,n̄)) for k = 1:N+1]  # add one to the end to use as an intermediate result

    E = TO.CostExpansion{T}(n̄,m,N)
    quad_exp = TO.CostExpansion(E, prob.model)
    Q = TO.CostExpansion{T}(n̄,m,N)
    Qprev = TO.CostExpansion{T}(n̄,m,N)
    S = TO.CostExpansion{T}(n̄,m,N)

    # Q_tmp = TO.QuadraticCost{T}(n̄,m)
    Q_tmp = TO.Expansion{T}(n̄,m)
	Quu_reg = SizedMatrix{m,m}(zeros(m,m))
	Qux_reg = SizedMatrix{m,n̄}(zeros(m,n̄))
    ρ = zeros(T,1)
    dρ = zeros(T,1)

    cache = FiniteDiff.JacobianCache(prob.model)
    exp_cache = TO.ExpansionCache(prob.obj)
    grad = zeros(T,N-1)

    logger = SolverLogging.default_logger(opts.verbose >= 2)
	L = typeof(prob.model)
	O = typeof(prob.obj)
    solver = iLQRSolver{T,QUAD,L,O,n,n̄,m,n+m,typeof(exp_cache)}(
        prob.model, prob.obj, x0, xf,
		prob.tf, N, opts, stats,
        Z, Z̄, K, d, D, G, quad_exp, S, E, Q, Qprev, Q_tmp, Quu_reg, Qux_reg, ρ, dρ, 
        cache, exp_cache, grad, logger)

    reset!(solver)
    return solver
end

# Getters
Base.size(solver::iLQRSolver{<:Any,<:Any,<:Any,<:Any,n,<:Any,m}) where {n,m} = n,m,solver.N
@inline TO.get_trajectory(solver::iLQRSolver) = solver.Z
@inline TO.get_objective(solver::iLQRSolver) = solver.obj
@inline TO.get_model(solver::iLQRSolver) = solver.model
@inline get_initial_state(solver::iLQRSolver) = solver.x0
@inline TO.integration(solver::iLQRSolver{<:Any,Q}) where Q = Q
solvername(::Type{<:iLQRSolver}) = :iLQR

log_level(::iLQRSolver) = InnerLoop

function reset!(solver::iLQRSolver{T}) where T
    reset_solver!(solver)
    solver.ρ[1] = 0.0
    solver.dρ[1] = 0.0
    return nothing
end

