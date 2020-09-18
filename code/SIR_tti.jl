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
function trajectory!(rng, list_time, list_I, system)
    rates = current_rates(system)
    pass_update!(rates, event) = update!(rates, event, system, rng)

    list_I[1] = system.measure_I
    time_simulation = Float64(list_time[1])
    for i in 2:length(list_time)
        if time_simulation < list_time[i]
            dtime = list_time[i] - time_simulation
            dtime_sim = advance!(KineticMonteCarlo(), rng, rates, pass_update!, dtime)
            time_simulation += dtime_sim
        end
        list_I[i] = system.measure_I
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
mutable struct SIR
    rng::AbstractRNG
    # rates
    epsilon::Float64 # rate of susceptible individual getting infected from outside
    mu::Float64      # rate of infected individual to recover
    lambda::Float64  # rate of infected individual to spread the disease
    nu::Float64      # rate of any individual to be in contact with anyone else
    omega_r::Float64 # rate of random test on any population individual
    omega_s::Float64 # rate of symptomatic-driven test on symptomatic
    # probabilities
    p_asymptomatic::Float64
    p_infection::Float64
    p_prevalence::Float64
    # compartments
    N::Int          # population size (S+I+R)
    S::Int          # susceptible
    I::Int          # infected (=H_a+H_s+T)
    H_a::Int        # hidden asymptomatic infected
    H_s::Int        # hidden symptomatic infected
    T::Int          # traced infected
    R::Int          # recovered
    measure_S::Int
    measure_I::Int
    measure_R::Int

    function SIR(rng::AbstractRNG, mu::Number, lambda::Number, nu::Number, omega_r::Number, omega_s::Number;
                 epsilon::Number = 0,
                 p_asymptomatic::Number = 0.5, # wise knowledge of Seba, reference will follow
                 p_prevalence::Number = 3e-3, # wise knowledge of Seba, reference will follow
                 N::Int = Int(1e4),
                )
        H_a = H_s = T = 0
        if rand(rng) < p_asymptomatic
            H_a = 1
        else
            H_s = 1
        end
        R0 = rand(rng, 0:2*N*p_prevalence)
        I0 = H_a + H_s + T
        S0 = N - I0 - R0
        epsilon = Float64(epsilon)
        mu      = Float64(mu)
        lambda  = Float64(lambda)
        nu      = Float64(nu)
        omega_r = Float64(omega_r)
        omega_s = Float64(omega_s)
        p_infection = lambda/nu

        new(rng, epsilon, mu, lambda, nu, omega_r, omega_s, p_asymptomatic, p_infection, p_prevalence, N, S0, I0, H_a, H_s, T, R0, S0, I0, R0)
    end
end

"""
update rates and system according to the last event that happend:
1: recovery
2: contact internal
3: infection external
4: random test
5: symptom-driven test
"""
function update!(rates::AbstractVector, event::Int, system::SIR)
    system.measure_S = system.S
    system.measure_I = system.I
    system.measure_R = system.R
    if index == 1 # recovery
        i = sample(system.rng, [1,2,3], AbstractWeights([system.H_a, system.H_s, system.T]./system.I))
        if i==1
            H_a -= 1
        elseif i==2
            H_s -= 1
        else
            T -= 1
        end
        system.R += 1
    elseif index == 2 # infection
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

"""
evaluate current rates of SIR system
"""
function current_rates(system::SIR)
    rate_recovery = system.mu * system.I
    rate_contact_internal   = system.nu * system.I
    rate_infection_external = system.epsilon * system.S
    # random tests can be restricted in simulation to only testing infected and
    # then simply to occur less often
    rate_test_random  = system.omega_r * (system.H_s + system.H_a)
    rate_test_symptom = system.omega_s * system.H_s
    return MVector{5,Float64}(rate_recovery, rate_contact_internal, rate_infection_external, rate_test_random, rate_test_symptom)
end







#function delete_from_system!(system, index)
#    update_sum!(system, -system.current_lambda[index])
#    deleteat!(system.current_lambda, index)
#end
#
#function add_to_system!(system, lambda)
#    push!(system.current_lambda, lambda)
#    update_sum!(system, lambda)
#end
#
#function update_sum!(system, change)
#    system.sum_current_lambda += change
#    system.update_current_lambda += 1
#    if system.update_current_lambda > 1e5
#        system.sum_current_lambda = sum(system.current_lambda)
#        system.update_current_lambda = 0
#    end
#end
