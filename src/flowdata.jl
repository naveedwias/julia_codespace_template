using Triangulate
using SimplexGridFactory

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