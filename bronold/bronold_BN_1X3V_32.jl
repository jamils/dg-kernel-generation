using Distributed
@everywhere using QuadGK
@everywhere using HCubature
@everywhere using FileIO

@everywhere using SharedArrays

# Basis Functions

@everywhere function basis(x, vx, vy, vz)
    [
        1/4,
        (sqrt(3) * x)/4,
        (sqrt(3) * vx)/4,
        (sqrt(3) * vy)/4,
        (sqrt(3) * vz)/4,
        (3 * vx * x)/4,
        (3 * vy * x)/4,
        (3 * vx * vy)/4,
        (3 * vz * x)/4,
        (3 * vx * vz)/4,
        (3 * vy * vz)/4,
        (3 * sqrt(5) * (x^2 - 1/3))/8,
        (3 * sqrt(5) * (vx^2 - 1/3))/8,
        (3 * sqrt(5) * (vy^2 - 1/3))/8,
        (3 * sqrt(5) * (vz^2 - 1/3))/8,
        (3^(3/2) * vx * vy * x)/4,
        (3^(3/2) * vx * vz * x)/4,
        (3^(3/2) * vy * vz * x)/4,
        (3^(3/2) * vx * vy * vz)/4,
        (3 * sqrt(15) * (vx * x^2 - vx/3))/8,
        (3 * sqrt(15) * (vx^2 * x - x/3))/8,
        (3 * sqrt(15) * (vy * x^2 - vy/3))/8,
        (3 * sqrt(15) * (vx^2 * vy - vy/3))/8,
        (3 * sqrt(15) * (vy^2 * x - x/3))/8,
        (3 * sqrt(15) * (vx * vy^2 - vx/3))/8,
        (3 * sqrt(15) * (vz * x^2 - vz/3))/8,
        (3 * sqrt(15) * (vx^2 * vz - vz/3))/8,
        (3 * sqrt(15) * (vy^2 * vz - vz/3))/8,
        (3 * sqrt(15) * (vz^2 * x - x/3))/8,
        (3 * sqrt(15) * (vx * vz^2 - vx/3))/8,
        (3 * sqrt(15) * (vy * vz^2 - vy/3))/8,
        (9 * vx * vy * vz * x)/4,
        (9 * sqrt(5) * (vx * vy * x^2 - (vx * vy)/3))/8,
        (9 * sqrt(5) * (vx^2 * vy * x - (vy * x)/3))/8,
        (9 * sqrt(5) * (vx * vy^2 * x - (vx * x)/3))/8,
        (9 * sqrt(5) * (vx * vz * x^2 - (vx * vz)/3))/8,
        (9 * sqrt(5) * (vx^2 * vz * x - (vz * x)/3))/8,
        (9 * sqrt(5) * (vy * vz * x^2 - (vy * vz)/3))/8,
        (9 * sqrt(5) * (vx^2 * vy * vz - (vy * vz)/3))/8,
        (9 * sqrt(5) * (vy^2 * vz * x - (vz * x)/3))/8,
        (9 * sqrt(5) * (vx * vy^2 * vz - (vx * vz)/3))/8,
        (9 * sqrt(5) * (vx * vz^2 * x - (vx * x)/3))/8,
        (9 * sqrt(5) * (vy * vz^2 * x - (vy * x)/3))/8,
        (9 * sqrt(5) * (vx * vy * vz^2 - (vx * vy)/3))/8,
        (9 * sqrt(15) * (vx * vy * vz * x^2 - (vx * vy * vz)/3))/8,
        (9 * sqrt(15) * (vx^2 * vy * vz * x - (vy * vz * x)/3))/8,
        (9 * sqrt(15) * (vx * vy^2 * vz * x - (vx * vz * x)/3))/8,
        (9 * sqrt(15) * (vx * vy * vz^2 * x - (vx * vy * x)/3))/8
    ];
end

@everywhere const numBasis = 48;

# Bronold & Fehske QM model
@everywhere const χ = 4.5;
@everywhere const mc = 0.26;

@everywhere function η(E, ξ)
    ans = sqrt(1 - ((E - χ) / (mc * E)) * (1 - ξ^2));
    return ans;
end

@everywhere function ξc(E)
    if E < (χ / (1-mc))
        ans = 0.0;
    else
        ans = sqrt(1 - ((mc * E) / (E - χ)));
    end
    return ans;
end

@everywhere function T(E, ξ)
    ans = (4 * mc * sqrt(E - χ) * ξ * sqrt(mc * E) * η(E,ξ)) / (mc * sqrt(E - χ) * ξ + sqrt(mc * E) * η(E,ξ))^2;
    return ans;
end

@everywhere function Tint(E)
    Tintans, err = quadgk(t -> T(E,t), ξc(E), 1);
    return Tintans
end

@everywhere function R(E, ξ, C)
    if E<χ
        Rans = 1;
    elseif ξ>ξc(E)
        Rans = 1 - (T(E, ξ) / (1 + (C/ξ))) - ((C/ξ) / (1 + (C/ξ))) * Tint(E);
    else
        Rans = 1 - ((C/ξ) / (1 + (C/ξ))) * Tint(E);
    end
    return Rans;
end

# Grid

@everywhere const elemCharge = 1.6021766 * 10^(-19);
@everywhere const me = 9.109383 * 10^(-31);
@everywhere const roughness = 2.0;

@everywhere vth = sqrt((10*elemCharge) / me);

@everywhere const Nv = 32; # 32

@everywhere dv = (12 * vth) / Nv;

@everywhere vc = zeros(Nv);

@everywhere for i in 1:Nv
    vc[i] = -6.0 * vth + (1/2 + (i-1)) * dv;
end

print(vc)

@everywhere function getE(vx, vy, vz, vcx, vcy, vcz, dv)
    (1/2) * me * ((vcx + vx *(dv/2))^2 + (vcy + vy * (dv/2))^2 + (vcz + vz * (dv/2))^2) / elemCharge;
end

@everywhere function getξ(vx, vy, vz, vcx, vcy, vcz, dv)
    (vcx + vx*(dv/2))/sqrt((vcx + vx*(dv/2))^2 + (vcy + vy*(dv/2))^2 + (vcz + vz*(dv/2))^2);
end

# Code generation

filename = "wall_BN_1X3V_32_"
println("Starting main loop") 

@time for j in 1:Nv
    io = open(filename*string(j)*".lua", "w");                                                            
    write(io, "return function (idx, f, out) \n");
    print("|",j,"| ")
    temp = SharedArray{Float64}(Nv, numBasis, numBasis); # temp[n, k, l]
    @time for m = 1:Nv
        print(m,", ")
        @sync @distributed for n = 1:Nv
            print(".")
            for k=1:numBasis
                for l = 1:numBasis
                    f(x, vx, vy, vz) = R(getE(vx, vy, vz, vc[j], vc[m], vc[n], dv), getξ(vx, vy, vz, vc[j], vc[m], vc[n], dv), roughness)*basis(-x, -vx, vy, vz)[l]*basis(x, vx, vy, vz)[k];
                    temp[n, k, l], err = hcubature(w -> f(w[1], w[2], w[3], w[4]), [-1,-1,-1,-1], [1,1,1,1]; maxevals = 9);
                end
            end
        end
    end
    write(io, string("   if idx[1] == ", j, " then \n"));
    for m = 1:Nv÷3
    	write(io, string("      if idx[2] == ", m, " then \n"));
        for n = 1:Nv
            write(io, string("         if idx[3] == ", n, " then \n"));
            for k = 1:numBasis
                str = "0.0";
                for l = 1:numBasis
                    if temp[n, k, l] ≠ 0.0
                        str *= string(" + ", temp[n, k, l], "*f[", l, "]");
                    end
                end
                write(io, string("            out[", k, "] = ", str, "\n"));
            end
            write(io, "         end \n");
        end
        write(io, "      end \n");
    end
    write(io, "   end \n");
    write(io, string("   if idx[1] == ", j, " then \n")); 
    for m = (Nv÷3+1):2*(Nv÷3)
    	write(io, string("      if idx[2] == ", m, " then \n"));
        for n = 1:Nv
            write(io, string("         if idx[3] == ", n, " then \n"));
            for k = 1:numBasis
                str = "0.0";
                for l = 1:numBasis
                    if temp[n, k, l] ≠ 0.0
                        str *= string(" + ", temp[n, k, l], "*f[", l, "]");
                    end
                end
                write(io, string("            out[", k, "] = ", str, "\n"));
            end
            write(io, "         end \n");
        end
	    write(io, "      end \n");
    end
    write(io, "   end \n");
    write(io, string("   if idx[1] == ", j, " then \n"));
    for m = (2*(Nv÷3)+1):Nv
    	write(io, string("      if idx[2] == ", m, " then \n"));
        for n = 1:Nv
            write(io, string("         if idx[3] == ", n, " then \n"));
            for k = 1:numBasis
                str = "0.0";
                for l = 1:numBasis
                    if temp[n, k, l] ≠ 0.0
                        str *= string(" + ", temp[n, k, l], "*f[", l, "]");
                    end
                end
                write(io, string("            out[", k, "] = ", str, "\n"));
            end
            write(io, "         end \n");
        end
        write(io, "      end \n");
    end
    write(io, "   end \n");
    write(io, "end\n");
    close(io);
end


println("Done!")