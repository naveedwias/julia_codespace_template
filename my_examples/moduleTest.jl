module Test
    # import Pkg; Pkg.add("Flux")
    using Flux
# import Pkg; Pkg.add("Zygote")
    using Zygote
# import Pkg; Pkg.add("Plots")
    using Plots
    using LinearAlgebra

    using Flux.Optimise: Adam

    using Statistics

    function build_model(num_layers, num_neurons)
        layers = Vector{Any}()  # Use a vector to store layers
        push!(layers, Dense(1, num_neurons, relu))  # Input layer with ReLU activation
        for _ in 1:(num_layers - 1)  # Hidden layers
            push!(layers, Dense(num_neurons, num_neurons, relu))
        end
        push!(layers, Dense(num_neurons, 1))  # Output layer without activation
        Chain(layers...)  # Convert the list of layers into a Chain model
    end
    println("done 2")
    
    # He initialization
    function he_initialization!(model)
        for layer in model
            if typeof(layer) <: Dense
                fan_in = size(layer.weight, 2)
                stddev = sqrt(2.0 / fan_in)
                layer.weight .= randn(size(layer.weight)) .* stddev
                layer.bias .= 0.0
            end
        end
    end
    println("done 3")

    # # Initialize the neural network
num_layers = 2
num_neurons = 10
model = build_model(num_layers, num_neurons)
he_initialization!(model)

# # Optimizer
learning_rate = 1e-4
opt = Adam(learning_rate)
println("done 4")
println("done")

function pde_residual(x, trial_p_func, k)
    # Compute the first derivative
    p_x = gradient(xi -> trial_p_func(xi), x)[1]
    
    # Compute the second derivative
    p_xx = gradient(xi -> gradient(xii -> trial_p_func(xii), xi)[1], x)[1]
    
    # Residual of the Helmholtz equation
    return p_xx + k^2 * trial_p_func(x)
end
println("done 6")
# Trial solution for scalar x
function trial_solution_scalar(x, nn_output, p1, p2, x1, x2)
    φ1 = (x - x1) / (x2 - x1)
    φ2 = (x2 - x) / (x2 - x1)
    φ2 * p1 + φ1 * p2 + φ1 * φ2 * nn_output
end
println("done 7")

# Training loop adjustments
function train_step!(x_train, model, opt)
    global function_evals = 0
    # Compute loss for all training points
    loss_fn(x_points) = mean([pde_residual(
                                x,
                                xi -> trial_solution_scalar(xi, model([xi])[1], p1, p2, x1, x2),
                                k
                              )^2 for x in x_points])

    # Compute gradients and update parameters
    grads = gradient(Flux.params(model)) do
        loss = loss_fn(x_train[1, :])
        global function_evals += 1
        loss
    end

    update!(opt, Flux.params(model), grads)
    return loss_fn(x_train[1, :]), grads
end

x1, x2 = 0.0, 1.0  # Domain boundaries
p1, p2 = 1.0, -1.0  # Boundary conditions
f = 2000  # Frequency in Hz
c = 340  # Speed of sound in m/s
ω = 2π * f
k = ω / c  # Wavenumber

# Training points
n_points = 100
num_epochs = 5000
loss_history = Float64[]
println("done 4")
x_train = LinRange(x1, x2, n_points) |> collect  # Generate 1D training points
x_train = reshape(x_train, 1, :)  # Reshape to (1, number of samples)

    # Initialize the training loop
loss_history = Float64[]
println("done 8")
num_epochs = 1000
for epoch in 1:num_epochs
    loss, grads = train_step!(x_train, model, opt)
    push!(loss_history, loss)
    println(epoch)

    if epoch % 2 == 0 || epoch == num_epochs
        feasibility = mean([abs(pde_residual(
                                   x,
                                   xi -> trial_solution_scalar(xi, model([xi])[1], p1, p2, x1, x2),
                                   k
                               )) for x in x_train[1, :]])
        first_order_opt = sum(norm(g) for g in grads)
        println("Epoch $epoch, Loss: $loss, Feasibility: $feasibility, First-Order Optimality: $first_order_opt, Function Evaluations: $function_evals")
    end
end
end