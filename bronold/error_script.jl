@everywhere using SharedArrays

@everywhere global dx = 0.0001;
@everywhere global n = 10000; #1 / dx
@everywhere global dt = 3.74999999e-6;
@everywhere global CFL = 0.75;

@everywhere runs = 3;
@everywhere div = [1, 2, 4];

Q0_shared = SharedArray{Float64}(8, n, runs);
Q_shared = SharedArray{Float64}(8, n, runs);

@sync @distributed for ri in 1:runs
    dx = dx * div[ri];
    n = n / div[ri];
    include("mhd_wave.jl")
    Q0_shared[:,:,ri] = Q0;
    Q_shared[:,:,ri] = Q;
end
