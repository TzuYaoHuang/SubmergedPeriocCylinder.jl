using WaterLily
using InterfaceAdvection
using StaticArrays
using Plots

"""
    submergedCylinder2D(N; Re=100, Fr=0.5, h=1, Hw=4, Ha=2, Lx=16, ...)

Rotating circular cylinder of diameter `D=N` cells beneath a flat free surface.
The fluid starts at rest; motion is driven only by the counter-clockwise rotation.

Non-dimensionalisation (length `D`, velocity `U = ωR`, the cylinder surface speed):
- `Re = U D / ν`
- `Fr = U / √(g D)`

Geometry (all in units of `D`):
- `h`: submergence depth, from the undisturbed free surface to the cylinder centre
- `Hw`: water depth (bottom wall to free surface)
- `Ha`: air layer height above the free surface
- `Lx`: domain length in `x` (periodic)
"""
function submergedCylinder2D(N; Re=100, Fr=0.5, h=1, Hw=4, Ha=2, Lx=16,
                             λμ=1e-2, λρ=1e-3, T=Float32, mem=Array, λ=Koren)
    D = N
    R = T(D/2)
    NN = (Lx*N, (Hw+Ha)*N)
    g = T(1)
    zeroT = zero(T)
    U = Fr*√(g*h*D) |> T
    ω = T(U/R)
    ν = U*D/Re |> T

    grav(i,x,t) = i==1 ? zeroT : -g

    # Free surface at y = Hw*D, water (dark) below
    yₛ = T(Hw*D)
    Inter(xyz) = xyz[2] - yₛ

    # Rotating cylinder: map to the body frame rotating with angle ωt
    center = SA{T}[Lx*D/2, yₛ-h*D]
    sdf(ξ,t) = √sum(abs2,ξ) - R
    function map(x,t)
        s,c = sincos(ω*t)
        ξ = x - center
        SA[c*ξ[1]+s*ξ[2], -s*ξ[1]+c*ξ[2]]
    end
    body = AutoBody(sdf,map)

    return TwoPhaseSimulation(
        NN, (0, 0), D;
        U, Δt=0.001, ν, InterfaceSDF=Inter, T, body, λμ, λρ, mem, g=grav, perdir=(1,), λ
    )
end

"""
    plotFrame(sim, d; ωmax=5, Δω=0.5)

Vorticity `ωD/U` with the free surface (`f=0.5`) and the cylinder outline.
`d` is a work array (same size as `sim.flow.p`) for the body distance.
Vorticity lives on the cell edge (`x = I-2`), `f` and the body distance on the cell centre (`x = I-1.5`).
The colour levels are centred on zero so that `ω=0` falls inside one white band.
"""
function plotFrame(sim, d; ωmax=5, Δω=0.5)
    D, t = sim.L, WaterLily.time(sim)
    σ = sim.flow.σ

    # vorticity on the edge, blanked inside the body
    @inside σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
    @inside d[I] = WaterLily.sdf(sim.body, loc(0,I,eltype(d)) .- 0.5f0, t)
    @inside σ[I] = d[I] < 0 ? zero(eltype(σ)) : clamp(σ[I], -ωmax, ωmax)

    # body distance on the centre
    @inside d[I] = WaterLily.sdf(sim.body, loc(0,I,eltype(d)), t)

    R = inside(sim.flow.p)
    xe, ye = (axes(σ[R],1) .- 1)/D, (axes(σ[R],2) .- 1)/D     # edge
    xc, yc = (axes(σ[R],1) .- 0.5)/D, (axes(σ[R],2) .- 0.5)/D # centre

    levels = range(-ωmax-Δω/2, ωmax+Δω/2; step=Δω)
    plt = contourf(xe, ye, σ[R]'|>Array; levels, color=:seismic, clims=extrema(levels), linewidth=0,
                   aspect_ratio=:equal, xlims=extrema(xc), ylims=extrema(yc), framestyle=:box,
                   xlabel="x/D", ylabel="y/D", colorbar_title="ωD/U", size=(1000,450), left_margin=5Plots.mm,
                   title="tU/D=$(round(sim_time(sim), digits=1))")
    contour!(plt, xc, yc, sim.intf.f[R]'|>Array; levels=[0.5], color=:black, lw=2, colorbar_entry=false)
    contour!(plt, xc, yc, d[R]'|>Array; levels=[0], color=:black, lw=1, colorbar_entry=false)
    return plt
end

# ---------------- run ----------------
N, Re, Fr = 16, 3000, 0.5
duration, step = 100, 0.1

sim = submergedCylinder2D(N; Re, Fr)
d = similar(sim.flow.p)
Plots.default(fontfamily="Computer Modern")

anim = @animate for tᵢ in range(0, duration; step)
    sim_step!(sim, tᵢ; verbose=false)
    plotFrame(sim, d)
    plot!(title="Re=$Re, Fr=$Fr, tU/D=$(round(tᵢ, digits=1))")
    println("tU/D=", round(tᵢ, digits=2), ", Δt=", round(sim.flow.Δt[end], digits=4))
end

mp4(anim, joinpath(@__DIR__, "..", "gfx", "2DSubmergedCylinder_Re$(Re)_Fr$(Fr).mp4"); fps=20)
