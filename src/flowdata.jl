using Triangulate
using SimplexGridFactory

function ConvQuad(n=10, α=1.0)
    ω = zeros(n+1)
    ω[1] = 1.0
    ω[2] = -α
    for j=3:n+1 
        ω[j] = (j-2-α)/(j-1) * ω[j-1]
    end
    s = zeros(n+1)
    s[1] = ω[1]
    for j=2:n+1 
        s[j] = s[j-1] + ω[j]
    end
    return ω, s
end

function convection!(result, nu, ∇u)
    result[1] += ∇u[1]*u[1] + ∇u[2]*u[2]
    result[2] += ∇u[3]*u[1] + ∇u[4]*u[2]
end

function get_grid(level; points = [0 0 ; 1 0 ; 1 1 ; 0 1], uniform = true)
    if uniform
    @info "Setting up uniform grid "
        xgrid = simplexgrid(Triangulate;
                    points=points',
                    bfaces=[1 2 ; 2 3 ; 3 4 ; 4 1 ]',
                    bfaceregions=[1, 2, 3, 4],
                    regionpoints=[0.5 0.5;]',
                    regionnumbers=[1],
                    regionvolumes=[0.5])
        xgrid = uniform_refine(xgrid,level)
    else
        xgrid = simplexgrid(Triangulate;
                    points=points',
                    bfaces=[1 2 ; 2 3 ; 3 4 ; 4 1 ]',
                    bfaceregions=[1, 2, 3, 4],
                    regionpoints=[0.5 0.5;]',
                    regionnumbers=[1],
                    regionvolumes=[4.0^(-level-1)*4])
    end
    return xgrid
end

function get_problemdata(problem; Re = 1, nonlinear = true, instationary = true)
    points = [0 0 ; 1 0 ; 1 1 ; 0 1]
    periodic = false
    μ = 1/Re
    if problem == 1
        u,dtu, p,∇u,f = get_flowdata_linear(μ, nonlinear, instationary)    
        @show "Linear space and time example!!!!"
    else
        @error "problem $problem not defined"
    end
    return points,u,dtu,p,∇u,f,μ, periodic
end

function get_flowdata_linear(ν, nonlinear, instationary)
    u = DataFunction((result, x, t) -> (
            result[1] = x[1]+x[2];
            result[2] = -(x[1]+x[2]);
            result .*= 3*t;
        ), [2,2]; name = "u", dependencies = "XT", bonus_quadorder = 1)
    p = DataFunction((result, x, t) -> (
            result[1] = 0
        ), [1,2]; name = "p", dependencies = "XT", bonus_quadorder = 0)
    ∇u = ∇(u)
    dtu = dt(u)
    
    f = DataFunction((result, x, t) -> (
            fill!(result,0);
            if nonlinear
                convection!(result, u(x,t), ∇u(x,t));
            end;
            if instationary
                result .+= dtu(x,t);
            end;
        ), [2,2]; name = "f", dependencies = "XT", bonus_quadorder = 1)
    return u, dtu, p, ∇u, f
end

function get_flow_data(ν)
    ## note that dependencies "XT" marks the function to be x- and t-dependent
    ## that causes the solver to automatically reassemble associated operators in each time step
    u = DataFunction((result, x, t) -> (
      result[1] =  2*x[1]*x[2]*cos(10 * π * t); 
      result[2] = -3*x[1]^2* sin(10 * π * t) - x[2]^2 * cos(10*π * t);
      ), [2,2]; name = "u", dependencies = "XT", bonus_quadorder = 5)
    u_t = DataFunction((result, x, t) -> (
      result[1] = -20*π*x[1]*x[2]*cos(10 * π * t); 
      result[2] = -30*π*x[1]^2* cos(10 * π * t) + 10*π*x[2]^2 * sin(10*π * t);
      ), [2,2]; name = "u", dependencies = "XT", bonus_quadorder = 5)
    p = DataFunction((result, x, t) -> (
  #       result[1] = -(x[1]^2 + x[2]^2 - 2.0/3.0)*(1.5 + 0.5 * t^(3/4))
      result[1] = -(x[1]^3 + x[2]^3 - 0.5)*(1.5 + 0.5 * sin(10 * π * t))
      ), [1,2]; name = "p", dependencies = "XT", bonus_quadorder = 5)
  
    ############## common code for all examples #####
    dt_u = eval_dt(u)
    Δu = eval_Δ(u)
    ∇p = eval_∇(p)
    f = DataFunction((result, x, t) -> (
          result .= dt_u(x,t) .- ν*Δu(x,t) .+ view(∇p(x,t),:);
        ), [2,2]; name = "f", dependencies = "XT", bonus_quadorder = 5)  
    return u, p, f, u_t
end

function fokker_plank_linear_space_time(ϵ)    
    function coeffbeta!(result, x, t)
        result[1] = x[1];
        result[2] = x[2];
    end    
    β = DataFunction(coeffbeta!, [2,2]; name="β", dependencies="XT", bonus_quadorder=1)
    function exact!(result, x, t)
        result[1] = x[1]*x[2]*(1.0+t);
    end    
    u = DataFunction(exact!, [1,2]; name="u", dependencies="XT", bonus_quadorder=5)    
    function rhs!(result, x, t)
        u = x[1]*x[2]*(1.0+t);
        ux = x[2]*(1.0+t);
        uxx = 0.0*(1.0+t);
        uy = x[2]*(1.0+t); 
        uyy = 0.0;
        ut = x[1]*x[2];
        result[1] = ut - ϵ*(uxx+uyy) + x[1]*ux+x[2]*uy + 2*u;
      return nothing
    end
    f = DataFunction(rhs!, [1,2]; name = "f", dependencies = "XT", bonus_quadorder = 5)  

    return β, u, ∇(u), f
end

function fokker_plank_space_example1(ϵ, p=1, α=1.0)
    γ= DataFunction([0.0]; name = "γ")    
    function coeffbeta!(result, x, t)
        result[1] = x[1];
        result[2] = x[2];
    end    
    β = DataFunction(coeffbeta!, [2,2]; name="β", dependencies="XT", bonus_quadorder=1)
    function exact!(result, x, t)
        result[1] = (x[1]-x[1]^2)*(x[2]-x[2]^2)*(1.0+t^α/gamma(α+1));
    end    
    u = DataFunction(exact!, [1,2]; name="u", dependencies="XT", bonus_quadorder=5)    
    ∇u = eval_∇(u)
    function rhs!(result, x, t)
        u = (x[1]-x[1]^2)*(x[2]-x[2]^2)*(1.0+t^α/gamma(α+1));
        ux = (1-2*x[1])*(x[2]-x[2]^2)*(1.0+t^α/gamma(α+1));
        uxx = -2*(x[2]-x[2]^2)*(1.0+t^α/gamma(α+1));
        uy = (x[1]-x[1]^2)*(1-2*x[2])*(1.0+t^α/gamma(α+1)); 
        uyy = -2*(x[1]-x[1]^2)*(1.0+t^α/gamma(α+1));
        ut = (x[1]-x[1]^2)*(x[2]-x[2]^2); # time derivative is 1
        result[1] = ut - (uxx+uyy) + x[1]*ux+x[2]*uy + 2*u;
      return nothing
    end
    f = DataFunction(rhs!, [1,2]; name = "f", dependencies = "XT", bonus_quadorder = 5)  

    return γ, β, u, ∇(u), f
end

function fokker_plank_space_notH2(ϵ, p=1, α=1.0)
    γ= DataFunction([0.0]; name = "γ")
    function coeffbeta!(result, x, t)
        result[1] = x[1];
        result[2] = x[2];
    end    
    β = DataFunction(coeffbeta!, [2,2]; name="β", dependencies="XT", bonus_quadorder=1)
    function exact!(result, x, t)
        exp_x = exp((x[1]-1)/ϵ );
        exp_eps = 1.0/(1-exp(-1.0/ϵ));
        u_0_x=x[1]-exp_eps*(exp_x-exp(-1.0/ϵ));
        result[1] = x[2]*(1-x[2])*u_0_x* (1.0+t^α/gamma(α+1));
    end    
    u = DataFunction(exact!, [1,2]; name="u", dependencies="XT", bonus_quadorder=5)
    function rhs!(result, x, t)
        exp_x = exp((x[1]-1)/ϵ );
        exp_eps = 1.0/(1-exp(-1.0/ϵ));
        u_0_x=x[1]-exp_eps*(exp_x-exp(-1.0/ϵ));
        u = (x[2]-x[2]^2)*u_0_x*(1.0+t^α/gamma(α+1));
        ux = x[2]*(1-x[2])*(1-(exp_x*exp_eps)/ϵ)*(1.0+t^α/gamma(α+1));        
        uy = (u_0_x*(1-2*x[2]))*(1.0+t^α/gamma(α+1));
        ut = (x[2]-x[2]^2)*u_0_x; # time derivative is 1
        laplace = ϵ*(-x[2]*(1-x[2])*(exp_x*exp_eps)/(ϵ^2)-2*u_0_x)*(1.0+t^α/gamma(α+1));
        result[1] = ut - laplace + x[1]*ux+x[2]*uy + 2*u;
      return nothing
    end
    f = DataFunction(rhs!, [1,2]; name = "f", dependencies = "XT", bonus_quadorder = 5)  

    return γ, β, u, ∇(u), f
end

function fokker_plank_space_nonsmooth1(ϵ, p=1, α=1.0)
    γ= DataFunction([0.0]; name = "γ")
    function coeffbeta!(result, x, t)
        result[1] = x[1];
        result[2] = x[2];
    end    
    β = DataFunction(coeffbeta!, [2,2]; name="β", dependencies="XT", bonus_quadorder=1)
    function exact!(result, x, t)
        result[1] = (x[1] >=0 && x[1] <= 0.5) ? 1.0 : 0.0;
    end
    u = DataFunction(exact!, [1,2]; name="u", dependencies="XT", bonus_quadorder=5)
    function rhs!(result, x, t)
        result[1] = 0.0;
      return nothing
    end
    f = DataFunction(rhs!, [1,2]; name = "f", dependencies = "XT", bonus_quadorder = 5)  

    return γ, β, u, ∇(u), f
end

function fokker_plank_space_example3(ϵ, p=1, α=1.0)
    γ= DataFunction([0.0]; name = "γ")    
    function coeffbeta!(result, x, t)
        result[1] = x[1];
        result[2] = x[2];
    end    
    β = DataFunction(coeffbeta!, [2,2]; name="β", dependencies="XT", bonus_quadorder=1)
    function exact!(result, x, t)
        result[1] = (x[1]-x[1]^2)*(x[2]-x[2]^2);
    end    
    u = DataFunction(exact!, [1,2]; name="u", dependencies="XT", bonus_quadorder=5)
    function rhs!(result, x, t)
        result[1] = 0.0;
      return nothing
    end
    f = DataFunction(rhs!, [1,2]; name = "f", dependencies = "XT", bonus_quadorder = 5)  

    return γ, β, u, ∇(u), f
end

using MittagLeffler
function fp_kassem_example1(ϵ=1.0, α=1.0, mm=2, nn=2)
    γ= DataFunction([0.0]; name = "γ")    
    function coeffbeta!(result, x, t)
        result[1] = x[1];
        result[2] = x[2];
    end    
    β = DataFunction(coeffbeta!, [2,2]; name="β", dependencies="XT", bonus_quadorder=1)

    function exact!(result, x, t)
        val = zero(Float64)
        for m=1:mm
            λm = (2*m-1)*π
            for n=1:nn
                λn = (2*n-1)*π
                λmn = ((2*m-1)^2 + (2*n-1)^2)*π^2
                val += (λm*λn)^(-3) * sin(λm * x[1]) * sin(λn * x[2]) * mittleff(α, 1.0, -λmn*t^α)
            end
        end
        result[1] = 64*val;
    end
    u = DataFunction(exact!, [1,2]; name="u", dependencies="XT", bonus_quadorder=5)

    function ut_exact!(result, x, t)
        val = zero(Float64)
        for m=1:mm
            λm = (2*m-1)*π
            for n=1:nn
                λn = (2*n-1)*π
                λmn = ((2*m-1)^2 + (2*n-1)^2)*π^2
                val += (λm*λn)^(-3) * sin(λm * x[1]) * sin(λn * x[2])# * mittleffderiv(α, α, -λmn*t^α)
            end
        end
        result[1] = 64*val;
    end
    ut = DataFunction(ut_exact!, [1,2]; name="u", dependencies="XT", bonus_quadorder=5)

    function fu!(result, x, t)
        exact!(result, x, t)
        result[2] = x[2]*result[1]
        result[1] = x[1]*result[1]
    end
    fu_data = DataFunction(fu!, [1,2]; name="u", dependencies="XT", bonus_quadorder=5)
    
    function initial!(result, x, t)
        result[1] = (x[1]-x[1]^2)*(x[2]-x[2]^2);
    end
    u0 = DataFunction(initial!, [1,2]; name="u", dependencies="XT", bonus_quadorder=5)
        
    function rhs!(result, x, t)
        val = zero(Float64)
        for m=1:mm
            λm = (2*m-1)*π
            for n=1:nn
                λn = (2*n-1)*π
                λmn = ((2*m-1)^2 + (2*n-1)^2)*π^2
                E_αα = mittleff(α, α, -λmn*t^α)
                term1 = (λm*x[1]*cos(λm * x[1]) + 2*sin(λm * x[1])) *sin(λn*x[2])
                term2 = λn*x[2]*sin(λm*x[1])*cos(λn*x[2])
                val +=  (λm*λn)^(-3) *( term1 + term2) * E_αα
            end
        end
        result[1] = 64*t^(α-1)*val;
      return nothing
    end
    f = DataFunction(rhs!, [1,2]; name = "f", dependencies = "XT", bonus_quadorder = 5)  

    # function rhs_auto!(result, x, t)
    #     diff_fu = div(fu_data)
    #     laplace = Δ(u)
    #     println(diff_fu)
    #     result[1] = ut(x,t)[1] - laplace(x,t)[1] + diff_fu(x,t)[1]
    #     return nothing
    # end
    # fn = DataFunction(rhs_auto!, [1,2]; name = "f", dependencies = "XT", bonus_quadorder = 5)  
    println("Kassem Example Series 1")
    return γ, β, u, ∇(u), f, u0
end


function fp_kassem_example2(ϵ=1.0, α=1.0, mm=2, nn=2)
    γ= DataFunction([0.0]; name = "γ")    
    function coeffbeta!(result, x, t)
        result[1] = x[1];
        result[2] = x[2];
    end    
    β = DataFunction(coeffbeta!, [2,2]; name="β", dependencies="XT", bonus_quadorder=1)

    
    function exact!(result, x, t)
        val = zero(Float64)
        for m=1:mm
            λm = (2*m-1)*π
            for n=1:nn
                λn = (2*n-1)*π
                λmn = ((2*m-1)^2 + (2*n-1)^2)*π^2
                val += (λm*λn)^(-3) * sin(λm * x[1]) * sin(λn * x[2]) * mittleff(α, 1.0, -λmn*t^α)
            end
        end
        result[1] = 64*val;
    end
    u = DataFunction(exact!, [1,2]; name="u", dependencies="XT", bonus_quadorder=5)

    function ut_exact!(result, x, t)
        val = zero(Float64)
        for m=1:mm
            λm = (2*m-1)*π
            for n=1:nn
                λn = (2*n-1)*π
                λmn = ((2*m-1)^2 + (2*n-1)^2)*π^2
                val += (λm*λn)^(-3) * sin(λm * x[1]) * sin(λn * x[2]) * mittleffderiv(α, 1, -λmn*t^α)
            end
        end
        result[1] = 64*val;
    end
    ut = DataFunction(ut_exact!, [1,2]; name="u", dependencies="XT", bonus_quadorder=5)

    function space_der_exact!(result, x, t)
        val = zero(Float64)
        for m=1:mm
            λm = (2*m-1)*π
            for n=1:nn
                λn = (2*n-1)*π
                λmn = ((2*m-1)^2 + (2*n-1)^2)*π^2
                val += (λm*λn)^(-3) * (2*sin(λm * x[1]) * sin(λn * x[2]) + λm*x[1]*cos(λm*x[1])*sin(λn*x[2]) + λn *x[2]*sin(λm*x[1])*cos(λn*x[2])) * mittleff(α, 1.0, -λmn*t^α)
            end
        end
        result[1] = 64*val;
    end
    dvi_uf = DataFunction(space_der_exact!, [1,2]; name="u", dependencies="XT", bonus_quadorder=5)
    
    function initial!(result, x, t)
        result[1] = (x[1]-x[1]^2)*(x[2]-x[2]^2);
    end
    u0 = DataFunction(initial!, [1,2]; name="u", dependencies="XT", bonus_quadorder=5)
        
    function rhs!(result, x, t)
        Δu = Δ(u)
        result[1] = ut(x,t)[1] - Δu(x,t)[1] + dvi_uf(x,t)[1]
      return nothing
    end
    f = DataFunction(rhs!, [1,2]; name = "f", dependencies = "XT", bonus_quadorder = 5)  
    println("Kassem Example Series 2")
    return γ, β, u, ∇(u), f, u0
end

function fp_kassem_example_nsmooth(nn=30, mm=30, α=1.0)
    γ= DataFunction([0.0]; name = "γ")
    function coeffbeta!(result, x, t)
        result[1] = x[1];
        result[2] = x[2];
    end    
    β = DataFunction(coeffbeta!, [2,2]; name="β", dependencies="XT", bonus_quadorder=5)
    function exact!(result, x, t)
        val = zero(Float64)
        for m=1:mm
            lm = (2*m-1)*π
            for n=1:mm
                ln = (2*n-1)*π
                lmn = ((2*m-1)^2 + (2*n-1)^2)*π^2
                pp = (m-1)*(n-1)
                val += (-1)^pp*(lm*ln)^(-2) * sin(lm * x[1]) * sin(ln * x[2]) * mittleff(α, 1.0, -lmn*t^α)
            end
        end
        result[1] = 16*val;
    end
    u = DataFunction(exact!, [1,2]; name="u", dependencies="XT", bonus_quadorder=5)

    function rhs!(result, x, t)
        val = zero(Float64)
        for m=1:mm
            lm = (2*m-1)*π
            for n=1:nn
                ln = (2*n-1)*π
                lmn = ((2*m-1)^2 + (2*n-1)^2)*π^2
                E_αα = mittleff(α, α, -lmn*t^α)
                term1 = (lm*x[1]*cos(lm * x[1]) + 2*sin(lm * x[1])) *sin(ln*x[2])
                term2 = ln*x[2]*sin(lm*x[1])*cos(ln*x[2])
                pp = (m-1)*(n-1)
                val +=  (-1)^pp*(lm*ln)^(-2) *( term1 + term2) * E_αα
            end
        end
        result[1] = 16*t^(α-1)*val;
      return nothing
    end
    f = DataFunction(rhs!, [1,2]; name = "f", dependencies = "XT", bonus_quadorder = 5)  

    function initial!(result, x, t)
        result[1] = (x[1] >0.5 && x[1] < 0.5) ? 1.0 : 0.0;
    end
    u0 = DataFunction(initial!, [1,2]; name="u", dependencies="XT", bonus_quadorder=5)
    
    return γ, β, u, ∇(u), f, u0
end