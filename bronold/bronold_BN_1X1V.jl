using Distributed
@everywhere using QuadGK
@everywhere using HCubature
@everywhere using FileIO

@everywhere using SharedArrays

# Basis Functions

@everywhere function basis(x, vx)
    [
        1/2,
	sqrt(3)*x/2,
	sqrt(3)*vx/2,
	3*vx*x/2,
	3*sqrt(5)*(x^2 - 1/3)/4,
	3*sqrt(5)*(vx^2 - 1/3)/4,
	3*sqrt(15)*(vx*x^2 - vx/3)/4,
	3*sqrt(15)*(vx^2*x - x/3)/4,
    ];
end

@everywhere const numBasis = 8;

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

@everywhere vth = sqrt((10*elemCharge)/me);

@everywhere const Nv = 32; # 32

@everywhere dv = (12*vth)/Nv;

@everywhere vc = zeros(Nv);

@everywhere for i in 1:Nv
    vc[i] = -6.0*vth + (1/2 + (i - 1))*dv;
end

@everywhere function getE(vx, vcx, dv)
    (1/2)*me*(vcx + vx *(dv/2))^2/elemCharge;
end

# Code generation

io = open("wall_BN_1X1V.lua", "w");

write(io, "local _M = {} \n");
write(io, "_M[1] = function (idx, f, out) \n");

println("Starting main loop")

temp = SharedArray{Float64}(Nv, numBasis, numBasis); # temp[n, k, l]
@sync @distributed for n = 1:Nv
  print(".")
  for k=1:numBasis
      for l = 1:numBasis
         f(x, vx) = R(getE(vx, vc[n], dv), 1.0, roughness) * basis(-x, -vx)[l] * basis(x, vx)[k];
         temp[n, k, l], err = hcubature(w -> f(w[1], w[2]), [-1, -1], [1, 1]; maxevals = 9);
      end
   end
end
for n = 1:Nv
    write(io, string("   if idx[1] == ", n, " then \n"));
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

write(io, "end \n");
write(io, "return _M \n");
close(io);


println("Done!")