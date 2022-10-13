using Distributed
@everywhere using QuadGK
@everywhere using HCubature
@everywhere using FileIO

@everywhere using SharedArrays

# Basis Functions

@everywhere function basis(x, vx, vy)
    [
        1/2^(3/2),
	sqrt(3)*x/2^(3/2),
	sqrt(3)*vx/2^(3/2),
	sqrt(3)*vy/2^(3/2),
	3*x*vx/2^(3/2),
	3*x*vy/2^(3/2),
	3*vx*vy/2^(3/2),
	3*sqrt(5)*(x^2 - 1/3)/2^(5/2),
	3*sqrt(5)*(vx^2 - 1/3)/2^(5/2),
	3*sqrt(5)*(vy^2 - 1/3)/2^(5/2),
	3*sqrt(3)*x*vx*vy/2^(3/2),
	sqrt(15)*vx*(3*x^2 - 1)/2^(5/2),
	sqrt(15)*vy*(3*x^2 - 1)/2^(5/2),
	sqrt(15)*x*(3*vx^2 - 1)/2^(5/2),
	sqrt(15)*vy*(3*vx^2 - 1)/2^(5/2),
	sqrt(15)*x*(3*vy^2 - 1)/2^(5/2),
	sqrt(15)*vx*(3*vy^2 - 1)/2^(5/2),
	3*sqrt(5)*vx*vy*(3*x^2 - 1)/2^(5/2),
	3*sqrt(5)*x*vy*(3*vx^2 - 1)/2^(5/2),
	3*sqrt(5)*x*vx*(3*vy^2 - 1)/2^(5/2)
    ];
end

@everywhere const numBasis = 20;

# Bronold & Fehske QM model
@everywhere const χ = 4.5;
@everywhere const mc = 0.26;

@everywhere function η(E, ξ)
    ans = sqrt(1 - ((E - χ)/(mc*E))*(1 - ξ^2));
    return ans;
end

@everywhere function ξc(E)
    if E < (χ/(1 - mc))
        ans = 0.0;
    else
        ans = sqrt(1 - ((mc*E)/(E - χ)));
    end
    return ans;
end

@everywhere function T(E, ξ)
    ans = (4*mc*sqrt(E - χ)*ξ*sqrt(mc*E)*η(E, ξ))/(mc*sqrt(E - χ)*ξ + sqrt(mc*E)*η(E, ξ))^2;
    return ans;
end

@everywhere function Tint(E)
    Tintans, err = quadgk(t -> T(E, t), ξc(E), 1);
    return Tintans
end

@everywhere function R(E, ξ, C)
    if E < χ
        Rans = 1;
    elseif ξ > ξc(E)
        Rans = 1 - (T(E, ξ)/(1 + (C/ξ))) - ((C/ξ)/(1 + (C/ξ)))*Tint(E);
    else
        Rans = 1 - ((C/ξ)/(1 + (C/ξ)))*Tint(E);
    end
    return Rans;
end

# Grid

@everywhere const elemCharge = 1.6021766*10^(-19);
@everywhere const me = 9.109383*10^(-31);
@everywhere const roughness = 2.0;

@everywhere const vth = sqrt((10*elemCharge)/me);

@everywhere const Nv = 32; # 32

@everywhere const dv = (12*vth)/Nv;

@everywhere vc = zeros(Nv);

@everywhere for i in 1:Nv
    vc[i] = -6.0*vth + (1/2 + (i-1))*dv;
end

@everywhere function getE(vx, vy, vcx, vcy, dv)
    (1/2)*me*((vcx + vx*(dv/2))^2 + (vcy + vy*(dv/2))^2)/elemCharge;
end

# Code generation

io = open("wall_BN_1X2V_jl_step9999.lua", "w");

write(io, "local _M = {} \n");
write(io, "_M[1] = function (idx, f, out) \n");

println("Starting main loop")

# error = SharedArray{Float64}(Nv, Nv, numBasis, numBasis);

#=

# ==================== TESTING CHAMBER ====================
using Random

error_array = zeros(Nv, Nv, numBasis, numBasis);

tol = 1e-6;
step = 10000;
println("rtol = ", tol, ", step = ", step)
println("m, n, k, l")

@time for m = 1:Nv
    for n = 1:Nv
        for k = 1:numBasis
            for l = 1:numBasis
                f(x, vx, vy) = R(getE(vx, vy, vc[m], vc[n], dv), 1.0, roughness) * basis(-x, -vx, vy)[l] * basis(x, vx, vy)[k];
                println(m, ", ", n, ", ", k, ", ", l)
                @time temp, error = hcubature(w -> f(w[1], w[2], w[3]), [-1, -1, -1], [1, 1, 1]; atol=tol, rtol=tol, maxevals=step);
                println("error = ", error)
                error_array[m, n, k, l] = error;
            end
        end
    end
end

println("Maximum error: ", maximum(error_array))

error("Stop!")

@time for i = 1:10000
    temp = zeros(Nv, numBasis, numBasis)
    m = rand(1:Nv);
    n = rand(1:Nv);
    k = rand(1:numBasis);
    l = rand(1:numBasis);
    f(x, vx, vy) = R(getE(vx, vy, vc[m], vc[n], dv), 1.0, roughness) * basis(-x, -vx, vy)[l] * basis(x, vx, vy)[k];
    println(m, ", ", n, ", ", k, ", ", l)
    @time temp[n, k, l], error = hcubature(w -> f(w[1], w[2], w[3]), [-1, -1, -1], [1, 1, 1]; atol=tol, rtol=tol, maxevals=step);
    println("error = ", error)
    error_array[m, n, k, l] = error;
end

println("Maximum error: ", maximum(error_array))

error("Stop!")

# ==================== STOP ====================

=#

error_array = SharedArray{Float64}(Nv, Nv, numBasis, numBasis);

const tol = 1e-6;
const evals = 20000;
println("tolerance = ", tol, ", max evals = ", evals)

@time for m = 1:Nv
    print(m, ", ")
    temp = SharedArray{Float64}(Nv, numBasis, numBasis); # temp[n, k, l]
    @time @sync @distributed for n = 1:Nv
        for k=1:numBasis
            for l = 1:numBasis
                f(x, vx, vy) = R(getE(vx, vy, vc[m], vc[n], dv), 1.0, roughness) * basis(-x, -vx, vy)[l] * basis(x, vx, vy)[k];
                temp[n, k, l], error = hcubature(w -> f(w[1], w[2], w[3]), [-1, -1, -1], [1, 1, 1]; maxevals=evals, rtol=tol, atol=tol);
                error_array[m, n, k, l] = error;
            end
        end
    end
    println("    Maximum error: ", maximum(error_array[m,:,:,:]))
    for n = 1:Nv
        write(io, string("   if idx[1] == ", m, " and idx[2] == ", n, " then \n"));
        for k = 1:numBasis
            str = "0.0";
            for l = 1:numBasis
                if temp[n, k, l] ≠ 0.0
                    str *= string(" + ", temp[n, k, l], "*f[", l, "]");
                end
            end
            write(io, string("      out[", k, "] = ", str, "\n"));
        end
        write(io, "   end \n");
    end
end

println("Total Maximum Error: ", maximum(error_array))

write(io, "end \n");
write(io, "return _M \n");
close(io);


println("Done!")