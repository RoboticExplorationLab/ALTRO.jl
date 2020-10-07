
"""
Calculates the optimal feedback gains K,d as well as the 2nd Order approximation of the
Cost-to-Go, using a backward Riccati-style recursion. (non-allocating)
"""
function backwardpass!(solver::iLQRSolver{T,QUAD,L,O,n,n̄,m}) where {T,QUAD<:QuadratureRule,L,O,n,n̄,m}
	N = solver.N

    # Objective
    obj = solver.obj
    model = solver.model

    # Extract variables
    Z = solver.Z; K = solver.K; d = solver.d;
    G = solver.G
    S = solver.S
	Quu_reg = solver.Quu_reg
	Qux_reg = solver.Qux_reg

    # Terminal cost-to-go
	Q = solver.Q[N]
    S[N].Q .= Q.Q
    S[N].q .= Q.q

    # Initialize expecte change in cost-to-go
    ΔV = @SVector zeros(2)

	k = N-1
    while k > 0
        ix = Z[k]._x
        iu = Z[k]._u

		# Get error state expanions
		fdx,fdu = TO.error_expansion(solver.D[k], model)
		cost_exp = solver.Q[k]
		Q = solver.Q_tmp

		# Calculate action-value expansion
		_calc_Q!(Q, cost_exp, S[k+1], fdx, fdu, S[k])

		# Regularization
		Quu_reg .= Q.R #+ solver.ρ[1]*I
		Quu_reg .+= solver.ρ[1]*Diagonal(@SVector ones(m))
		Qux_reg .= Q.H

	    if solver.opts.bp_reg
	        vals = eigvals(Hermitian(Quu_reg))
	        if minimum(vals) <= 0
	            @warn "Backward pass regularized"
	            regularization_update!(solver, :increase)
	            k = N-1
	            ΔV = @SVector zeros(2)
	            continue
	        end
	    end

        # Compute gains
		_calc_gains!(K[k], d[k], Quu_reg, Qux_reg, Q.r)

		# Calculate cost-to-go (using unregularized Quu and Qux)
		ΔV += _calc_ctg!(S[k], Q, K[k], d[k])

        k -= 1
    end

    regularization_update!(solver, :decrease)

    return ΔV

end

function static_backwardpass!(solver::iLQRSolver{T,QUAD,L,O,n,n̄,m}) where {T,QUAD<:QuadratureRule,L,O,n,n̄,m}
	N = solver.N

    # Objective
    obj = solver.obj
    model = solver.model

    # Extract variables
    Z = solver.Z; K = solver.K; d = solver.d;
    G = solver.G
    S = solver.S
	Quu_reg = SMatrix(solver.Quu_reg)
	Qux_reg = SMatrix(solver.Qux_reg)

    # Terminal cost-to-go
	# Q = error_expansion(solver.Q[N], model)
	Q = solver.Q[N]
	Sxx = SMatrix(Q.Q)
	Sx = SVector(Q.q)

    # Initialize expected change in cost-to-go
    ΔV = @SVector zeros(2)

	k = N-1
    while k > 0
        ix = Z[k]._x
        iu = Z[k]._u

		# Get error state expanions
		fdx,fdu = TO.error_expansion(solver.D[k], model)
		fdx,fdu = SMatrix(fdx), SMatrix(fdu)
		Q = TO.static_expansion(solver.Q[k])
		# Q = error_expansion(solver.Q[k], model)
		# Q = solver.Q[k]

		# Calculate action-value expansion
		Q = _calc_Q!(Q, Sxx, Sx, fdx, fdu)

		# Regularization
		Quu_reg, Qux_reg = _bp_reg!(Q, fdx, fdu, solver.ρ[1], solver.opts.bp_reg_type)

	    if solver.opts.bp_reg
	        vals = eigvals(Hermitian(Quu_reg))
	        if minimum(vals) <= 0
	            @warn "Backward pass regularized"
	            regularization_update!(solver, :increase)
	            k = N-1
	            ΔV = @SVector zeros(2)
	            continue
	        end
	    end

        # Compute gains
		K_, d_ = _calc_gains!(K[k], d[k], Quu_reg, Qux_reg, Q.u)

		# Calculate cost-to-go (using unregularized Quu and Qux)
		Sxx, Sx, ΔV_ = _calc_ctg!(Q, K_, d_)
		# k >= N-2 && println(diag(Sxx))
		if solver.opts.save_S
			S[k].Q .= Sxx
			S[k].q .= Sx
			S[k].c = sum(ΔV_)
		end
		ΔV += ΔV_
        k -= 1
    end

    regularization_update!(solver, :decrease)

    return ΔV
end

function _bp_reg!(Quu_reg::SizedMatrix{m,m}, Qux_reg, Q, fdx, fdu, ρ, ver=:control) where {m}
    if ver == :state
        Quu_reg .= Q.uu #+ solver.ρ[1]*fdu'fdu
		mul!(Quu_reg, Transpose(fdu), fdu, ρ, 1.0)
        Qux_reg .= Q.ux #+ solver.ρ[1]*fdu'fdx
		mul!(Qux_reg, fdu', fdx, ρ, 1.0)
    elseif ver == :control
        Quu_reg .= Q.uu #+ solver.ρ[1]*I
		Quu_reg .+= ρ*Diagonal(@SVector ones(m))
        Qux_reg .= Q.ux
    end
end

function _bp_reg!(Q, fdx, fdu, ρ, ver=:control)
    if ver == :state
		Quu_reg = Q.uu + ρ * fdu'fdu
		Qux_reg = Q.ux + ρ * fdu'fdx
    elseif ver == :control
		Quu_reg = Q.uu + ρ * I
        Qux_reg = Q.ux
    end

	Quu_reg, Qux_reg
end

function _calc_Q!(Q, cost_exp, S1, fdx, fdu, Q_tmp)
	# Compute the cost-to-go, stashing temporary variables in S[k]
	# Qx =  Q.x[k] + fdx'S.x[k+1]
	mul!(Q.q, Transpose(fdx), S1.q)
	Q.q .+= cost_exp.q

    # Qu =  Q.u[k] + fdu'S.x[k+1]
	mul!(Q.r, Transpose(fdu), S1.q)
	Q.r .+= cost_exp.r

    # Qxx = Q.xx[k] + fdx'S.xx[k+1]*fdx
	mul!(Q_tmp.Q, Transpose(fdx), S1.Q)
	mul!(Q.Q, Q_tmp.Q, fdx)
	Q.Q .+= cost_exp.Q

    # Quu = Q.uu[k] + fdu'S.xx[k+1]*fdu
	mul!(Q_tmp.H, Transpose(fdu), S1.Q)
	mul!(Q.R, Q_tmp.H, fdu)
	Q.R .+= cost_exp.R

    # Qux = Q.ux[k] + fdu'S.xx[k+1]*fdx
	mul!(Q_tmp.H, Transpose(fdu), S1.Q)
	mul!(Q.H, Q_tmp.H, fdx)
	Q.H .+= cost_exp.H

	return nothing
end

function _calc_Q!(Q::TO.StaticExpansion, Sxx, Sx, fdx::SMatrix, fdu::SMatrix)
	Qx = Q.x + fdx'Sx
	Qu = Q.u + fdu'Sx
	Qxx = Q.xx + fdx'Sxx*fdx
	Quu = Q.uu + fdu'Sxx*fdu
	Qux = Q.ux + fdu'Sxx*fdx
	TO.StaticExpansion(Qx,Qxx,Qu,Quu,Qux)
end


function _calc_gains!(K::SizedArray, d::SizedArray, Quu::SizedArray, Qux::SizedArray, Qu)
	LAPACK.potrf!('U',Quu.data)
	K .= Qux
	d .= Qu
	LAPACK.potrs!('U', Quu.data, K.data)
	LAPACK.potrs!('U', Quu.data, d.data)
	K .*= -1
	d .*= -1
	# return K,d
end

function _calc_gains!(K, d, Quu::SMatrix, Qux::SMatrix, Qu::SVector)
	K_ = -Quu\Qux
	d_ = -Quu\Qu
	K .= K_
	d .= d_
	return K_,d_
end

function _calc_ctg!(S, Q, K, d)
	# S.x[k]  =  Qx + K[k]'*Quu*d[k] + K[k]'* Qu + Qux'd[k]
	tmp1 = S.r
	S.q .= Q.q
	mul!(tmp1, Q.R, d)
	mul!(S.q, Transpose(K), tmp1, 1.0, 1.0)
	mul!(S.q, Transpose(K), Q.r, 1.0, 1.0)
	mul!(S.q, Transpose(Q.H), d, 1.0, 1.0)

	# S.xx[k] = Qxx + K[k]'*Quu*K[k] + K[k]'*Qux + Qux'K[k]
	tmp2 = S.H
	S.Q .= Q.Q
	mul!(tmp2, Q.R, K)
	mul!(S.Q, Transpose(K), tmp2, 1.0, 1.0)
	mul!(S.Q, Transpose(K), Q.H, 1.0, 1.0)
	mul!(S.Q, Transpose(Q.H), K, 1.0, 1.0)
	transpose!(Q.Q, S.Q)
	S.Q .+= Q.Q
	S.Q .*= 0.5

    # calculated change is cost-to-go over entire trajectory
	t1 = dot(d, Q.r)
	mul!(Q.r, Q.R, d)
	t2 = 0.5*dot(d, Q.r)
    return @SVector [t1, t2]
end

function _calc_ctg!(Q::TO.StaticExpansion, K::SMatrix, d::SVector)
	Sx = Q.x + K'Q.uu*d + K'Q.u + Q.ux'd
	Sxx = Q.xx + K'Q.uu*K + K'Q.ux + Q.ux'K
	Sxx = 0.5*(Sxx + Sxx')
	# S.x .= Sx
	# S.xx .= Sxx
	t1 = d'Q.u
	t2 = 0.5*d'Q.uu*d
	return Sxx, Sx, @SVector [t1, t2]
end
