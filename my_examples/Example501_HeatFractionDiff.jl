module Example501_HeatFractionDiff
  include("../src/variationaltimedisc.jl")
  using LinearAlgebra
  using Printf
  using Jacobi

  using GradientRobustMultiPhysics
  using ExtendableGrids
  using GridVisualize
  using ExtendableSparse
  using SpecialFunctions
  
  function get_aj(j, beta)
    if j==0
      aj = (j+1)^(1-beta)
    else
      aj = (j+1)^(1-beta) - (j)^(1-beta)
    end
    return aj
  end

  # b1 = a1; b_{n+1}=-a_n, b_{n+1-i} = a_{n+1-i} - a_{n-i}
  function get_bi(n, beta)
    aj = zeros(Float64, n)
    bnmi = zeros(Float64, n)
    bnmi[1] = get_aj(0, beta)
    for i = 1 : n-1
        bnmi[n-i] = get_aj(n-i, beta) - get_aj(n-i-1, beta)
    end
    bnmi[n] = -get_aj(n-1, beta)
    return bnmi
  end


  function ReactionConvectionDiffusionOperator(α, β, ϵ)
    function action_kernel!(result, input, x, t)
        β.x = x
        β.time = t
        eval_data!( α )
        eval_data!( β )
        # α * u_h + β_1 * ∇_xu_h + β_2 ∇_y u_h
        result[1] = α.val[1] * input[1] + β.val[1] * input[2] + β.val[2] * input[3]
        # Laplacian
        result[2] = ϵ * input[2]
        result[3] = ϵ * input[3]
        return nothing
    end
    action = Action(action_kernel!, [3, 3], dependencies = "XT", bonus_quadorder = max(α.bonus_quadorder, β.bonus_quadorder))
    return BilinearForm([OperatorPair{Identity, Gradient}, OperatorPair{Identity, Gradient}], action;
    name=" ϵ(∇ u, ∇ v) + (α u + β⋅∇u, v)", transposed_assembly = true)
  end

  function get_problem_data(ν, beta=1)
    α = DataFunction([1.0]; name = "α")
    β = DataFunction([1.0,1.0]; name = "β")    
    function exact!(result, x, t)
      #result[1] = t^alpha*x[1]^2*(x[1]-1.0)^2*x[2]*(x[2]-1)^2 #sin(2*π*x[1])*sin(2*π*x[2])
      result[1] = (t^3 + t^2 +1) * (x[1]^2-1)^2*(x[2]^2-1)^2
     end
    
    u = DataFunction(exact!, [1,2]; name="u", dependencies="XT", bonus_quadorder=5)
    dt_u = eval_dt(u)
    ∇u = eval_∇(u)
    Δu = eval_Δ(u)
    function rhs!(result, x, t)
      # dt_u = gamma(3)*t^1.5/gamma(2.5)*x[1]#*(x[1]-1.0)*x[2]*(x[2]-1);#sin(2*π*x[1])*sin(2*π*x[2]);
      # temp = gamma(alpha+1)/gamma(alpha-beta+1)*t^(alpha-beta)*x[1];
      #temp = gamma(alpha+1)/gamma(alpha-beta+1)*t^(alpha-beta)*x[1]^2*(x[1]-1.0)^2*x[2]*(x[2]-1)^2;
      #dt_u = temp;
      temp = gamma(4)/gamma(4-beta) * t^(3-beta) + gamma(3)/gamma(3-beta)*t^(2-beta) ;
      temp = (3*t^2+2*t);
      dt_u = temp * (x[1]^2-1)^2*(x[2]-1)^2;
      result[1] = dt_u-ν*Δu(x,t)[1] + dot(β(), ∇u(x,t))[1] + dot(α(), u(x,t)[1]) # α * u(x,t)[1]
      return nothing
    end
    f = DataFunction(rhs!, [1,2]; name = "f", dependencies = "XT", bonus_quadorder = 5)
    return α, β, u, ∇(u), f
  end
  

  function main(; scheme = 1, ϵ = 1, verbosity = 0, nlevels=2, 
        T0 = 0, end_time=1, beta=1, nsteps=10)
    ## set log level
    set_verbosity(verbosity)
    ## load initial mesh
    xgrid = grid_unitsquare(Triangle2D)
    # choose a finite element type
    FEType = H1Pk{1,2,2}
    #TODO: fix from the problem data
    u0 = DataFunction([0.0])
    
    ## negotiate data functions to the package
    α, β, u, ∇u, f = get_problem_data(ϵ, beta)
    
    ## creating PDE description
    Problem = PDEDescription("Convection-Diffusion-Reaction problem")
    add_unknown!(Problem, unknown_name = "u", equation_name="Convection-Diffusion-Reaction problem")        
    add_operator!(Problem, [1,1], ReactionConvectionDiffusionOperator(α, β, ϵ))
    add_rhsdata!(Problem, 1, LinearForm(Identity, f))
    add_boundarydata!(Problem, 1, [1,2,3,4], BestapproxDirichletBoundary; data=u)

    FES = FESpace{FEType}(xgrid)
    for level = 1 : nlevels
      # refine the grid 
      xgrid = uniform_refine(xgrid)

      # generate FE spaces
      FES = FESpace{FEType}(xgrid)
    end
    # @show FES
    Solution = FEVector(FES)

    n_dofs = FES.ndofs
    interpolate!(Solution[1], u; time = 0.0)    

    M = FEMatrix(FES)
    assemble_operator!(M[1,1], BilinearForm([Identity, Identity]))
    # @show M.entries
    # println(size(M[1,1]))
    println("ndofs: ", FES.ndofs)

    A = FEMatrix(FES)
    assemble_operator!(A[1,1], ReactionConvectionDiffusionOperator(α, β, ϵ); time=0.0)
    # @show A.entries

    rhs = FEVector(FES)
    assemble_operator!(rhs[1], LinearForm(Identity, f); time=0.0)
    # @show rhs.entries

    dt = Array{BoundaryData,1}(undef,0)
    push!(dt, BoundaryData(BestapproxDirichletBoundary; regions = [1,2,3,4], data = u))
    dofs = boundarydata!(Solution[1], dt; time = 0.0)

    for dof in dofs
      A.entries[dof, dof] = 1e60
      rhs.entries[dof] = 1e60*Solution.entries[dof]
    end

    t0 = T0
    tau = (end_time - T0)/nsteps
    
    V1 = zeros(Float64, FES.ndofs, 1)
    Mu0 = zeros(Float64, FES.ndofs)    

    SystemMatrix = FEMatrix(FES)
    # @show SystemMatrix
    SystemRHS = FEVector(FES)
    SystemSol = FEVector(FES)
    
    eL2 = zero(Float64)
    eH1 = zero(Float64)
    oldL2 = zero(Float64); oldH1=zero(Float64)

    # beta = 1.0
    step = zero(Int)

    SolVector = Array{FEVector{Float64}}([])
    push!(SolVector, Solution)

    l2max = -one(Float64)
    for current_time = 1 : nsteps
      @printf("Time step: %d: [%.5f, %.5f]\n", current_time, t0, t0+tau)
      
      step += 1
      bmni = get_bi(step, beta)
      beta0 = gamma(2-beta)*tau^beta
      println("beta0 ", beta0)

      fill!(rhs.entries, 0)
      assemble_operator!(rhs[1], LinearForm(Identity, f), time = t0 + tau)
      # println(rhs[1])
      V1[:, 1] = rhs.entries
      # println("V1: ", V1)
      
      # println(typeof(Mu0))

      fill!(SystemRHS.entries, 0)      
      if scheme == 1
        Mu0[:] = M.entries*Solution[1].entries
        addblock!(SystemRHS[1], Mu0; factor= 1.0)
        addblock!(SystemRHS[1], V1[:,1]; factor= tau )
        # RHS = τ * F^n + M * U^{n-1}
      end      
      if scheme == 2
        count = zero(Float64)
        for i = 1 : step
          count -= bmni[step-i+1]
          # M * M^{i}
          # println("bmni[",step-i+1,"]=", bmni[step-i+1])          
          Mu0[:] = M.entries*SolVector[i].entries
          addblock!(SystemRHS[1], Mu0; factor= -bmni[step-i+1])
          # -b_{n-1} * M * U^i
        end
        addblock!(SystemRHS[1], V1[:,1]; factor= beta0 )
        # RHS = β_0 * F^n - ∑_{i=1}^n b_{n-i} * M * U^i
        println("sum of bis = ", count)
      end

      # reset the system matrix
      fill!(SystemMatrix.entries.cscmatrix.nzval, 0)
      fill!(A.entries.cscmatrix.nzval, 0)
      assemble_operator!(A[1, 1], ReactionConvectionDiffusionOperator(α, β, ϵ); time=t0 + tau )
      addblock!(SystemMatrix[1, 1], M[1, 1]; factor= 1.0)
      if scheme == 1
        addblock!(SystemMatrix[1, 1], A[1, 1]; factor= tau)
        # LHS = M + τ A 
      end
      if scheme == 2
        addblock!(SystemMatrix[1, 1], A[1, 1]; factor= beta0)
        # LHS = M + β_0 * A 
      end
      # boundary dofs correction
      dofs = boundarydata!(SystemSol[1], dt; time = t0 + tau)
      for dof in dofs
        SystemRHS[1][dof] = 1e60 * SystemSol[1][dof]
        SystemMatrix[1,1][dof,dof] = 1e60
      end
      # solve the system 
      flush!(SystemMatrix.entries)      
      #@show SystemRHS.entries
      
      SystemSol.entries[:] = SystemMatrix.entries \ SystemRHS.entries
      
      for j = 1 : length(Solution.entries)
        Solution[1][j] = SystemSol[1][j]
      end
      #interpolate!(Solution[1], u; time = t0+tau)
      # error computation
      H1Error = L2ErrorIntegrator(∇(u), Gradient; time=t0+tau)
      h1 = evaluate(H1Error,   Solution[1])
      eH1 += ( h1 + oldH1) * tau * 0.5
      oldH1 = h1
      L2Error_u = L2ErrorIntegrator(u, Identity; time= t0+tau )
      l2 = evaluate(L2Error_u, Solution[1])
      eL2 += (l2 + oldL2) * tau * 0.5
      oldL2 = l2
      l2max = max(l2max, l2)
      println("L2 error: ", sqrt(l2), " H1 error: ", sqrt(h1))

      for j = 1 : length(Solution.entries)
        Solution[1][j] = SystemSol[1][j]
      end
      push!(SolVector, Solution)
      t0 = t0 + tau
    end # endfor nsteps
    println("L2(0,t,L2): ", sqrt(eL2))
    println("L2(0,t,H1): ", sqrt(eH1))
    println("L_inf(L2): ", sqrt(l2max))
  end #end Main Function
end