using MonteCarloX
using Random
using StatsBase
using Printf
using ProgressMeter
using LinearAlgebra
using DelimitedFiles
using StaticArrays

"""
# generate a stochastic trajectory
SIR dynamics is goverened by differential equation
``
    \\frac{dI}{dt} = \\sum_{i=1}^{I}\\lambda_i\\frac{S}{N} - \\mu I
``

# Arguments
* distribution for the heterogeneous rates P(lambda)
"""
function trajectory!(rng, list_T, list_S, list_I, list_R, system)
    rates = current_rates(system)
    pass_update!(rates, index) = update!(rates, index, system, rng)
    
    list_S[1] = system.measure_S
    list_I[1] = system.measure_I
    list_R[1] = system.measure_R
    time_simulation = Float64(list_T[1])
    for i in 2:length(list_T)
        if time_simulation < list_T[i]
            dT = list_T[i] - time_simulation
            dT_sim = advance!(KineticMonteCarlo(), rng, rates, pass_update!, dT)
            time_simulation += dT_sim
        end
        list_S[i] = system.measure_S
        list_I[i] = system.measure_I
        list_R[i] = system.measure_R
    end
end

"""
Ingreedients:
- the rate of each agent to become infected or to recover
-> the rate of infected to make a connection with any other agent (well-mixed),
   which adds to the connection matrix, and only infects with a certain
   probability
   !! Connection matrix needs to be deleted after 2 weeks -> LightGraphs plus a
   queue where we store the edges and the time when stored. (clean the graph whenever it gets updated)

- the rate to be tested for those that will develop symptoms (probability)
    -> as soon as S->I in update we draw a random number, and if the id is symptom based then add this rate to the system
- the rate to be tested randomly
-> upon positive test: add contacts to a list of tracing
- the rate to test people from tracing list
"""
mutable struct SIR{D}
    epsilon::Float64
    mu::Float64
    P_lambda::D
    current_lambda::Vector{Float64}
    sum_current_lambda::Float64
    update_current_lambda::Int
    N::Int
    S::Int
    I::Int
    R::Int
    measure_S::Int
    measure_I::Int
    measure_R::Int
    
    function SIR{D}(rng::AbstractRNG, P_lambda::D, epsilon::Number, mu::Number, S0::Int, I0::Int, R0::Int) where D
        N = S0 + I0 + R0
        current_lambda = rand(rng, P_lambda, I0)
        sum_current_lambda = sum(current_lambda)
        new(Float64(epsilon), Float64(mu), P_lambda, current_lambda, sum_current_lambda, 0, N, S0, I0, R0, S0, I0, R0)
    end
end

"""
update rates and system according to the last event that happend:
index -> event type and agent id
e.g.: index = 1-10000 would mean update agent i out of 10.000 = N
      index = 10001 -> would mean to random test
      index = 10002++ -> would mean do a test from some list that we need to define
"""
function update!(rates::AbstractVector, index::Int, system::SIR, rng::AbstractRNG)
    system.measure_S = system.S
    system.measure_I = system.I
    system.measure_R = system.R
    if index == 1 # recovery
        random_I = rand(rng, 1:system.I)
        delete_from_system!(system, random_I)
        system.I -= 1
        system.R += 1
    elseif index == 2 # infection
        add_to_system!(system, rand(rng, system.P_lambda))
        system.S -= 1
        system.I += 1
        @assert length(system.current_lambda) == system.I
    elseif index == 0 # absorbing state of zero infected
        system.measure_S = system.S
        system.measure_I = system.I
        system.measure_R = system.R
    else
        throw(UndefVarError(:index))
    end
    
    rates .= current_rates(system)
end

function delete_from_system!(system, index)
    update_sum!(system, -system.current_lambda[index])
    deleteat!(system.current_lambda, index)
end

function add_to_system!(system, lambda)
    push!(system.current_lambda, lambda)
    update_sum!(system, lambda)
end

function update_sum!(system, change) 
    system.sum_current_lambda += change 
    system.update_current_lambda += 1
    if system.update_current_lambda > 1e5
        system.sum_current_lambda = sum(system.current_lambda)
        system.update_current_lambda = 0
    end
end

"""
evaluate current rates of SIR system
"""
function current_rates(system::SIR)
    rate_recovery = system.mu * system.I
    rate_infection = system.sum_current_lambda* system.S/system.N + system.epsilon
    return MVector{2,Float64}(rate_recovery, rate_infection)
end
