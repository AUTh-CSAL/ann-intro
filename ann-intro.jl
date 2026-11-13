try; import KaimonSlate; catch; error("This is a Kaimon Slate notebook — running it as plain Julia needs the KaimonSlate runtime in this environment. Add it with `import Pkg; Pkg.add(\"KaimonSlate\")`, or open it in Kaimon Slate."); end; KaimonSlate.standalone!(@__MODULE__; dir=@__DIR__)

#%% md id=intro
@md"""
# From functions to neural networks

A neural network is a function with adjustable **[parameters](#dict-parameters)** (the *decision variables*, or unknowns, of an optimization problem). **[Training](#dict-training)** it (*solving the optimization problem*) means finding parameter values that make a **[loss function](#dict-loss)** (the *objective*, or cost function) small.

This notebook is a bridge between two vocabularies: machine learning (ML) and numerical optimization. Each idea is introduced as **ML term** (*optimization term*), so you can read papers from either side. Every bold term links to a row of the two-way [dictionary](#dictionary) that closes the notebook, and each row links back to the section that introduces it.

By the end you will have derived by hand, and implemented from scratch in Julia, **[backpropagation](#dict-backpropagation)** (*reverse-mode differentiation*) for a two-layer network, trained it with **[stochastic gradient descent](#dict-sgd)** (a *stochastic gradient method*), and tested it on linearly separable data, XOR, circles and two kinds of spiral.

The path there:

- [A function can make a decision](#sec-classifier) (a *classifier*).
- [Minimizing a function](#sec-minimum): the *minimizer*.
- [Two parameters](#sec-landscape): the **[loss landscape](#dict-loss-landscape)** (*objective surface*) and its *level sets*.
- [The gradient](#sec-gradient) and the condition ∇F = 0 (*[stationarity](#dict-vanishing-gradient)*).
- [Gradient descent](#sec-descent) (*steepest descent*) and the **[learning rate](#dict-learning-rate)** (*step size*).
  - [An ill-conditioned valley](#sec-banana): the *banana* (*Rosenbrock*) function.
  - [A lattice of local minima](#sec-local-minima) (**[traps](#dict-local-minima)**): the banana times a 2-D sine, and the [basins of attraction](#sec-basins) that sort the starting points.
- [Stochastic gradient descent](#sec-sgd) on **mini-batches** (*sampled gradients*).
- [A neuron](#sec-neuron), a loss, and the chain rule.
- [Two layers and backpropagation](#sec-backpropagation).
- [Training on datasets](#sec-training) of increasing difficulty.
- [Experiments](#sec-experiments) and the [dictionary](#dictionary).

Controls change the mathematics, not just the appearance of the plots. Cells marked ✏️ are exercises with a built-in checker.
"""

#%% code id=setup hidecode
using CairoMakie

# Figures are shown at roughly their nominal size, so 1 pixel per unit is plenty
# (the default of 2 makes every redraw ~4× heavier for no visible gain).
CairoMakie.activate!(type = "png", px_per_unit = 1)
set_theme!(theme_dark())

"Ready"

#%% code id=shared_helpers hidecode
# Shared helpers, defined once. Packages are imported (not `using`), so their names
# stay qualified (Random.MersenneTwister, Statistics.mean) and do not leak into the notebook.
import Random, Statistics

# Sigmoid (logistic function)
σ(z) = 1 / (1 + exp(-z))

# Cross-entropy for one example (a = predicted probability, y = label)
ce(a, y) = -(y * log(a) + (1 - y) * log(1 - a));

#%% md id=classifier
@md raw"""
<a id="sec-classifier"></a>

## A function that makes a decision (a classifier)

We observe points described by two **[features](#dict-features)** (*input variables*) \(x_1,x_2\), each belonging to one of two **classes** (*labels*). We want a rule that predicts the class.

A simple rule is the **score** (a *linear model*)

\[
s(x_1,x_2)=w_1x_1+w_2x_2+b,
\]

with **weights** (*coefficients*) \(w_1,w_2\) and **bias** (*offset*, or intercept) \(b\). Together \(w_1,w_2,b\) are the **[parameters](#dict-parameters)** (*decision variables*). We predict class 1 when \(s\geq 0\), and class 0 otherwise.

The **[decision boundary](#dict-decision-boundary)** (the *zero level set* of \(s\)) is the line

\[
w_1x_1+w_2x_2+b=0.
\]

The weights set the line's direction (they are its normal vector) and the bias shifts it away from the origin. Move the parameters and watch the boundary change. The orange rings mark observations the current rule classifies incorrectly.

Counting those rings is the **[0–1 loss](#dict-zero-one-loss)** (the *misclassification count*). It is a poor objective for optimization: it only takes integer values, so it is flat almost everywhere and its gradient is zero. The [neuron](#sec-neuron) section replaces it with a smooth surrogate.
"""

#%% code id=classifier_data hidecode
# A small, fixed, linearly separable data set.
x1 = [-1.5, -1.1, -0.8, -0.4, -1.4, -0.6,
       0.5,  0.9,  1.2,  1.5,  0.7,  1.4]
x2 = [-1.0, -0.5, -1.3, -0.8, -1.5, -0.3,
       0.8,  1.3,  0.6,  1.1,  1.5,  0.4]

labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

#%% code id=classifier_controls collapsed hidecode
@bind w1 Slider(-2.0:0.1:2.0; label="w₁")
@bind w2 Slider(-2.0:0.1:2.0; label="w₂")
@bind bias Slider(-2.0:0.1:2.0; label="b")

#%% code id=classifier_plot controls=w1,w2,bias hidecode
xs = range(-2, 2; length=100)
ys = range(-2, 2; length=100)

# scores[j, i] is the score at (xs[i], ys[j]); Makie wants Z[x_index, y_index], hence the transpose below.
scores = [w1*x + w2*y + bias for y in ys, x in xs]

fig1 = Figure(size=(720, 480))
ax1 = Axis(fig1[1, 1];
    title="Linear decision rule",
    xlabel="x₁", ylabel="x₂",
    aspect=DataAspect())

# Faint shading of the region the rule assigns to class 1 (s ≥ 0).
heatmap!(ax1, xs, ys, (scores .>= 0)';
    colormap=[(:dodgerblue, 0.12), (:tomato, 0.12)], colorrange=(0, 1))

# The decision boundary is the zero contour.
if abs(w1) + abs(w2) > 1e-8
    contour!(ax1, xs, ys, scores';
        levels=[0.0], color=:white, linewidth=2)
end

# Show the original classes.
scatter!(ax1, x1[labels .== 0], x2[labels .== 0];
    color=:dodgerblue, markersize=13, label="Class 0")
scatter!(ax1, x1[labels .== 1], x2[labels .== 1];
    color=:tomato, markersize=13, label="Class 1")

# Highlight mistakes made by the current rule.
predictions = [
    (w1*x1[i] + w2*x2[i] + bias >= 0) ? 1 : 0
    for i in eachindex(labels)
]
wrong = findall(predictions .!= labels)

# Always plotted (even if empty) so the legend explains the orange ring.
scatter!(ax1, x1[wrong], x2[wrong];
    color=:transparent, strokecolor=:orange,
    strokewidth=2.5, markersize=23, label="Misclassified")

axislegend(ax1; position=:lt)
fig1

#%% code id=classifier_summary hidecode
(; w1, w2, bias, errors=length(wrong), total=length(labels))

#%% md id=minimum
@md raw"""
<a id="sec-minimum"></a>

## Finding a minimum (minimizing an objective)

Now consider a function of one variable, the **[loss](#dict-loss)** (*objective function*)

\[
f(x)=(x-1.2)^2.
\]

Its **minimizer** (the *argmin*) is \(x^\star=1.2\), where the minimum value is \(f(x^\star)=0\). Note the difference: the *minimizer* is the input \(x^\star\), the *minimum* is the value \(f(x^\star)\).

Move the slider. The marker is the point where we evaluate the function (in optimization language, the current **iterate**). The graph itself does not move.

This is the basic shape of an optimization problem, \(\min_x f(x)\): find the input that makes the function as small as possible. In ML terms, find the parameter value that makes the loss smallest. That search is what ML calls [training](#dict-training).
"""

#%% code id=minimum_controls hidecode
@bind x0 Slider(-2.0:0.05:3.0; default=-1.0, label="x")

#%% code id=minimum_plot controls=x0 hidecode
f1(x) = (x - 1.2)^2

xrange = range(-2, 3; length=400)
yrange = f1.(xrange)

fig2 = Figure(size=(720, 400))
ax2 = Axis(fig2[1, 1];
    title="A function and a selected point",
    xlabel="x", ylabel="f(x)")

lines!(ax2, xrange, yrange; color=:cyan, linewidth=2)
scatter!(ax2, [x0], [f1(x0)];
    color=:orange, markersize=16)

fig2

#%% code id=minimum_value hidecode
(; x=x0, f_x=f1(x0))

#%% md id=landscape
@md raw"""
<a id="sec-landscape"></a>

## A function of two parameters (the loss landscape)

A function can depend on more than one variable. Consider

\[
F(a,b)=(a-1)^2+1.5(b+0.5)^2
       +0.35(a-1)(b+0.5).
\]

Think of \(a\) and \(b\) as the **[parameters](#dict-parameters)** (*decision variables*) we are free to adjust.

The surface is the **[loss landscape](#dict-loss-landscape)** (the *graph of the objective*). The contour plot shows its **level sets**: curves along which \(F\) has the same value. Where the contours crowd together the surface is steep; where they spread apart it is flat.

Move the two sliders. The point moves over the surface and across the level sets.

The minimizer is \((a,b)=(1,-0.5)\). Next, derivatives will tell us which direction reduces \(F\).
"""

#%% code id=landscape_defs hidecode
F(a, b) = (a - 1)^2 + 1.5*(b + 0.5)^2 +
          0.35*(a - 1)*(b + 0.5)

# Analytic gradient (∂F/∂a, ∂F/∂b), used by the plots and by gradient descent.
# Exercise 1 asks you to derive the same thing yourself; `numgrad` is the independent check.
∇F(a, b) = (2*(a - 1) + 0.35*(b + 0.5),
            3*(b + 0.5) + 0.35*(a - 1))

avals = range(-2, 3; length=70)
bvals = range(-2, 2; length=70)
# Makie expects Z[a_index, b_index]
Z = [F(u, v) for u in avals, v in bvals];

#%% code id=landscape_controls hidecode
@bind a Slider(-2.0:0.1:3.0; default=-1.0, label="a")
@bind b Slider(-2.0:0.1:2.0; default=1.0, label="b")

#%% code id=landscape_plot controls=a,b hidecode
fig3 = Figure(size=(900, 420))

ax3 = Axis3(fig3[1, 1];
    title="The objective surface",
    xlabel="a", ylabel="b", zlabel="F(a,b)")

surface!(ax3, avals, bvals, Z)
scatter!(ax3, [a], [b], [F(a,b)];
    color=:orange, markersize=18)

ax4 = Axis(fig3[1, 2];
    title="Contours of the same function",
    xlabel="a", ylabel="b",
    aspect=DataAspect())

contour!(ax4, avals, bvals, Z;
    levels=12, linewidth=1.5)
scatter!(ax4, [a], [b];
    color=:orange, markersize=16)
scatter!(ax4, [1.0], [-0.5];
    color=:white, marker=:xcross, markersize=16)

fig3

#%% code id=landscape_value hidecode
(; a, b, F=F(a,b))

#%% code id=numgrad_def hidecode
# Central finite differences: (F(a+h) - F(a-h)) / 2h
function numgrad(f, a, b; h = 1e-5)
    ((f(a + h, b) - f(a - h, b)) / 2h,
     (f(a, b + h) - f(a, b - h)) / 2h)
end;

#%% md id=gradient
@md raw"""
<a id="sec-gradient"></a>

## Which way is downhill? The gradient

At a point \((a,b)\) we ask: if I nudge \(a\) a little, how fast does \(F\) change? That rate is the **partial derivative** \(\partial F/\partial a\) (treat \(b\) as constant). Stacking both partials gives the **gradient**

\[
\nabla F(a,b)=\begin{bmatrix}\partial F/\partial a\\[2pt] \partial F/\partial b\end{bmatrix}.
\]

(ML texts often write the same object as \(\partial L/\partial\theta\).) The gradient points in the direction of **steepest ascent**, perpendicular to the level set through the point. So \(-\nabla F\) is the **steepest descent direction**: the way we want to step.

At a minimizer the gradient vanishes, \(\nabla F=0\). This is the **first-order necessary condition** (*[stationarity](#dict-vanishing-gradient)*). It is only necessary: a maximum or a saddle point also has \(\nabla F=0\). Move the point below: the white arrow is \(\nabla F\), the orange arrow is \(-\nabla F\), and both shrink as you approach the minimizer.

The arrows here, and every gradient descent run from now on, use the **analytic** gradient: the formula obtained by differentiating by hand. A **finite-difference** approximation \((F(a+h,b)-F(a-h,b))/2h\) needs more function evaluations per step and is only approximate, so we keep it for one job: checking a formula.

**Exercise: the gradient.** Differentiate \(F\) by hand and fill in `∇F_mine` in the next cell. The checker compares your answer with a finite-difference approximation. ML practitioners call this test a **[gradient check](#dict-gradient-check)**, and we will reuse it to debug [backpropagation](#sec-backpropagation).
"""

#%% code id=gradient_exercise
# ✏️ Exercise: the gradient — replace the zeros with your hand-derived partial derivatives.
# F(a,b) = (a-1)^2 + 1.5(b+0.5)^2 + 0.35(a-1)(b+0.5)
function ∇F_mine(a, b)
    dFda = 0.0   # TODO
    dFdb = 0.0   # TODO
    return (dFda, dFdb)
end;

#%% code id=gradient_exercise_check hidecode
test_points = [(0.0, 0.0), (2.0, -1.0), (-1.0, 1.5), (1.0, -0.5)]
errs = [maximum(abs.(collect(∇F_mine(p...)) .- collect(numgrad(F, p...)))) for p in test_points]

if all(errs .< 1e-4)
    "✅ Correct — your gradient matches finite differences at all test points."
else
    "❌ Not yet. Max error per test point: $(round.(errs; digits=3)). Hint: differentiate each term, treating the other variable as a constant."
end

#%% code id=gradient_controls hidecode
@bind pa Slider(-2.0:0.1:3.0; default=-1.0, label="a")
@bind pb Slider(-2.0:0.1:2.0; default=1.0, label="b")

#%% code id=gradient_plot controls=pa,pb hidecode
gF = ∇F(pa, pb)               # (∂F/∂a, ∂F/∂b), analytic
gnorm = sqrt(gF[1]^2 + gF[2]^2)
sc = 0.8 / max(gnorm, 0.8)     # draw long arrows at a readable length

fig4 = Figure(size=(720, 480))
ax5 = Axis(fig4[1, 1]; title="Gradient (white) and steepest descent (orange)",
           xlabel="a", ylabel="b", aspect=DataAspect())
contour!(ax5, avals, bvals, Z; levels=14, linewidth=1.2)
scatter!(ax5, [1.0], [-0.5]; color=:white, marker=:xcross, markersize=16)
arrows2d!(ax5, [pa], [pb], [sc*gF[1]], [sc*gF[2]]; color=:white, shaftwidth=3, tipwidth=14, tiplength=14)
arrows2d!(ax5, [pa], [pb], [-sc*gF[1]], [-sc*gF[2]]; color=:orange, shaftwidth=3, tipwidth=14, tiplength=14)
scatter!(ax5, [pa], [pb]; color=:orange, markersize=14)
fig4

#%% md id=descent
@md raw"""
<a id="sec-descent"></a>

## Taking steps: gradient descent (steepest descent)

Repeat one rule: step against the gradient,

\[
\theta_{k+1}=\theta_k-\eta\,\nabla F(\theta_k),\qquad \theta=(a,b).
\]

The number \(\eta>0\) is the **[learning rate](#dict-learning-rate)** (the *step size*, or step length). Here it is fixed for all iterations. Optimization texts often choose it at each iteration by a *line search*; ML practice usually fixes it or decays it on a schedule.

- Too small: progress is safe but slow.
- Too large: we overshoot the valley, bounce around, and may [diverge](#dict-divergence).

Choose a **starting point** (the *initial guess* \(\theta_0\), called the **[initialization](#dict-initialization)** in ML) and a learning rate. The left plot shows the iterates on the level sets. The right plot shows \(F\) at each iteration, the **[training curve](#dict-training-curve)** (the *convergence history*). A straight line on this log scale means **linear convergence** (a constant error-reduction factor per step). Using every data point at each step, as here, is also called [full-batch](#dict-full-batch) gradient descent.

**Exercise: the stable step size.** The **Hessian** (the matrix of second derivatives, i.e. the *curvature*) of this \(F\) is
\(H=\begin{bmatrix}2&0.35\\0.35&3\end{bmatrix}\), with eigenvalues about 1.89 and 3.11. For a quadratic, fixed-step gradient descent converges only if \(\eta<2/\lambda_{\max}\approx 0.64\).

1. Use the slider to find the largest \(\eta\) that still converges, and compare it with 0.64.
2. Along an eigen-direction with curvature \(\lambda\), the error is multiplied by \((1-\eta\lambda)\) each step. Use this to explain the threshold.
3. Why does the ratio \(\lambda_{\max}/\lambda_{\min}\) (the **condition number**) govern how slow the best fixed step is?
"""

#%% code id=descent_method
# Gradient descent: θ ← θ − η ∇f(θ). `∇f(x, y)` returns the analytic gradient as a tuple.
# With `tol > 0` we also stop as soon as ‖∇f‖ ≤ tol (a stationary point has been reached).
function gradient_descent(∇f, θ0, η, n; tol = 0.0)
    path = [θ0]
    θ = θ0
    for _ in 1:n
        g = ∇f(θ...)
        hypot(g...) <= tol && break                       # stationary: stop
        θ = (θ[1] - η*g[1], θ[2] - η*g[2])
        push!(path, θ)
        (abs(θ[1]) > 1e6 || abs(θ[2]) > 1e6) && break   # diverged
    end
    path
end;

#%% code id=descent_controls hidecode
@bind gd_a0 Slider(-2.0:0.1:3.0; default=-1.5, label="start a (initial guess)")
@bind gd_b0 Slider(-2.0:0.1:2.0; default=1.5, label="start b")
@bind gd_lr Slider(0.01:0.01:0.8; default=0.1, label="learning rate (step size) η")
@bind gd_steps Slider(1:100; default=30, label="iterations")

#%% code id=descent_plot controls=[gd_a0,gd_b0],[gd_lr,gd_steps] hidecode
gd_path = gradient_descent(∇F, (gd_a0, gd_b0), gd_lr, gd_steps)
gd_vals = [clamp(F(p...), 1e-12, 1e12) for p in gd_path]

fig5 = Figure(size=(960, 440))
ax6 = Axis(fig5[1, 1]; title="Path on the contours", xlabel="a", ylabel="b", aspect=DataAspect())
contour!(ax6, avals, bvals, Z; levels=14, linewidth=1.0)
lines!(ax6, first.(gd_path), last.(gd_path); color=:orange, linewidth=1.5)
scatter!(ax6, first.(gd_path), last.(gd_path); color=:orange, markersize=7)
scatter!(ax6, [gd_a0], [gd_b0]; color=:lime, markersize=14)
scatter!(ax6, [1.0], [-0.5]; color=:white, marker=:xcross, markersize=16)
xlims!(ax6, -2, 3); ylims!(ax6, -2, 2)

ax7 = Axis(fig5[1, 2]; title="F after each step (log scale)", xlabel="step k", ylabel="F", yscale=log10)
lines!(ax7, 0:length(gd_vals)-1, gd_vals; color=:cyan, linewidth=2)
scatter!(ax7, 0:length(gd_vals)-1, gd_vals; color=:cyan, markersize=6)
fig5

#%% md id=banana
@md raw"""
<a id="sec-banana"></a>

## A harder valley: the banana function

The classical test problem of numerical optimization is the **banana function** (the *Rosenbrock function*)

\[
f(x,y)=(1-x)^2+b\,(y-x^2)^2,\qquad b=100.
\]

Its minimizer is \((1,1)\) with \(f=0\), at the bottom of a narrow, curved valley \(y=x^2\). The valley floor is nearly flat but its walls are very steep. At the minimizer the Hessian (curvature) has eigenvalues about \(1001.6\) and \(0.40\): a **condition number** (the ratio \(\lambda_{\max}/\lambda_{\min}\)) of about 2500. Such a problem is **ill-conditioned**; ML people say the loss has [*ravines*](#dict-ravines).

The gradient is two short lines, and the plots below use exactly this formula:

\[
\frac{\partial f}{\partial x}=-2(1-x)-4b\,x\,(y-x^2),\qquad
\frac{\partial f}{\partial y}=2b\,(y-x^2).
\]

Fixed-step gradient descent is squeezed between two demands. The step must satisfy \(\eta<2/\lambda_{\max}\), or it bounces across the valley (see the stable-step exercise in [gradient descent](#sec-descent)). But progress *along* the valley floor is governed by \(\lambda_{\min}\), so it crawls. Try \(\eta=0.001\), \(0.0019\) and \(0.0021\), and compare 5 000 with 20 000 iterations.

Remedies use curvature information (*Newton* and *quasi-Newton* methods, *preconditioning*) or [*momentum*](#dict-momentum) (the heavy-ball method).
"""

#%% code id=banana_defs
# The banana (Rosenbrock) function: minimizer (1, 1), f = 0, at the bottom of the valley y = x².
banana(x, y; b = 100.0) = (1 - x)^2 + b * (y - x^2)^2

# Its analytic gradient (the two lines the text derives).
∇banana(x, y; b = 100.0) = (-2*(1 - x) - 4*b*x*(y - x^2),
                            2*b*(y - x^2))

# Grid for the contour plot (log scale: the walls are far higher than the floor).
bn_x = range(-2, 2; length=200)
bn_y = range(-1, 3; length=200)
BZ = [log10(1 + banana(x, y)) for x in bn_x, y in bn_y];   # BZ[x_index, y_index]

#%% code id=banana_controls hidecode
@bind bn_x0 Slider(-2.0:0.1:2.0; default=-1.2, label="start x (initial guess)")
@bind bn_y0 Slider(-1.0:0.1:3.0; default=1.0, label="start y")
@bind bn_lr Slider(0.0002:0.0001:0.003; default=0.001, label="learning rate (step size) η")
@bind bn_steps Slider(500:500:20000; default=5000, label="iterations")

#%% code id=banana_plot controls=[bn_x0,bn_y0],[bn_lr,bn_steps] hidecode
bn_path = gradient_descent(∇banana, (bn_x0, bn_y0), bn_lr, bn_steps)
bn_vals = [clamp(banana(p...), 1e-16, 1e12) for p in bn_path]
bn_idx = 1:max(1, length(bn_path) ÷ 400):length(bn_path)    # thin the DRAWING, not the iteration

fig5b = Figure(size=(960, 440))
ax_b1 = Axis(fig5b[1, 1]; title="Gradient descent in the banana valley", xlabel="x", ylabel="y")
contour!(ax_b1, bn_x, bn_y, BZ; levels=20, linewidth=1.0)
lines!(ax_b1, first.(bn_path[bn_idx]), last.(bn_path[bn_idx]); color=:orange, linewidth=1.5)
scatter!(ax_b1, [bn_x0], [bn_y0]; color=:lime, markersize=14)
scatter!(ax_b1, [1.0], [1.0]; color=:white, marker=:xcross, markersize=16)
xlims!(ax_b1, -2, 2); ylims!(ax_b1, -1, 3)

ax_b2 = Axis(fig5b[1, 2]; title="f at each iteration (log scale)", xlabel="iteration k", ylabel="f", yscale=log10)
lines!(ax_b2, (0:length(bn_vals)-1)[bn_idx], bn_vals[bn_idx]; color=:cyan, linewidth=2)
fig5b

#%% md id=banana_3d
@md raw"""
**The same surface in 3-D.** Use the *azimuth* and *elevation* sliders to walk around the landscape. The height is \(\log_{10}(1+f)\), because the valley walls are thousands of times higher than the floor. The orange line is the path of gradient descent from the plot above, lifted onto the surface. From the side you can see why a fixed step struggles: the floor is long, shallow and curved, with steep walls on both sides. Look along the valley (try different azimuths) to see how flat the floor is.
"""

#%% code id=banana_3d_controls hidecode
@bind bd_azim Slider(0:5:360; default=230, label="azimuth (turn around) °")
@bind bd_elev Slider(5:5:85; default=30, label="elevation (look down) °")

#%% code id=banana_3d_plot controls=bd_azim,bd_elev hidecode
# The same banana surface in 3-D, drawn as height = log₁₀(1 + f) so the valley floor stays visible.
# Reads the path from the plot above, so moving the viewpoint never re-runs gradient descent.
b3_x = bn_x[1:2:end];  b3_y = bn_y[1:2:end];  b3_z = BZ[1:2:end, 1:2:end]
b3_path = bn_path[bn_idx]
b3_pz = [log10(1 + banana(p...)) + 0.08 for p in b3_path]      # lift the path slightly above the surface

fig5b3d = Figure(size=(900, 560))
ax_3d = Axis3(fig5b3d[1, 1]; title="The banana surface (height = log₁₀(1 + f))",
    xlabel="x", ylabel="y", zlabel="log₁₀(1 + f)",
    azimuth=deg2rad(bd_azim), elevation=deg2rad(bd_elev), aspect=(1, 1, 0.7))
surface!(ax_3d, b3_x, b3_y, b3_z; colormap=:viridis)
lines!(ax_3d, first.(b3_path), last.(b3_path), b3_pz; color=:orange, linewidth=3)
scatter!(ax_3d, [bn_x0], [bn_y0], [log10(1 + banana(bn_x0, bn_y0)) + 0.08]; color=:lime, markersize=14)
scatter!(ax_3d, [1.0], [1.0], [0.08]; color=:white, marker=:xcross, markersize=18)
fig5b3d

#%% code id=stepsize_exercise
# ✏️ Exercise: the stable step size — for b = 100 the Hessian of the banana function at its minimizer (1, 1) is
#       H = [ 802  -400 ]
#           [-400   200 ]
# Find its largest eigenvalue by hand (use the trace and the determinant), then predict the
# largest stable fixed step size η = 2/λ_max. Test your prediction with the slider above.
η_max_mine = 0.0   # TODO

#%% code id=stepsize_exercise_check hidecode
let tr = 802 + 200, dt = 802 * 200 - 400^2
    η_true = 2 / ((tr + sqrt(tr^2 - 4dt)) / 2)
    if abs(η_max_mine - η_true) / η_true < 0.02
        "✅ Correct to within 2%. Now check it: η just below your value converges, just above it does not settle."
    else
        "❌ Not yet. Hint: the eigenvalues solve λ² − (trace)λ + (determinant) = 0, and the stability threshold is 2 divided by the larger one."
    end
end

#%% md id=local_minima
@md raw"""
<a id="sec-local-minima"></a>

## A grid of local minima

Now make it harder. Multiply the banana (with a gentler \(b=10\)) by a 2-D sine ripple:

\[
g(x,y)=f(x,y)\,\big(1+A\sin(\omega x)\sin(\omega y)\big),\qquad 0\le A<1.
\]

The factor is always positive, so \(g\ge 0\) and \(g(1,1)=0\): the point \((1,1)\) stays the **global minimizer**. But the ripple adds local minima (ML: [*bad local minima*](#dict-local-minima); also *traps*): points where \(\nabla g=0\) and the curvature is positive, yet \(g>0\). The amplitude \(A\) sets how deep the traps are. We start with a slow ripple, **one period** across the plot window (\(\omega=\pi/2\)), which gives just a few traps; try two periods (\(\omega=\pi\)) to see the lattice fill in. Because the ripple is multiplied by \(f\), it grows with distance from the minimizer.

Fixed-step gradient descent now follows the nearest downhill path and stops in whichever trap it reaches first: the result depends on the **starting point** (the [initialization](#dict-initialization)). The plot below shows one run from the chosen start. The next part, on [basins of attraction](#sec-basins), runs many starts at once.
"""

#%% code id=local_minima_defs
# The banana times a 2-D sine ripple. A = 0 gives the plain banana; A < 1 keeps g ≥ 0, so (1, 1) stays the global minimizer.
banana_sine(x, y; A = 0.5, ω = 3.0, b = 10.0) = banana(x, y; b) * (1 + A * sin(ω * x) * sin(ω * y))

# Gradient by the product rule: ∇(f·s) = s ∇f + f ∇s, with s = 1 + A sin(ωx) sin(ωy).
function ∇banana_sine(x, y; A = 0.5, ω = 3.0, b = 10.0)
    s  = 1 + A * sin(ω * x) * sin(ω * y)
    gf = ∇banana(x, y; b)
    f  = banana(x, y; b)
    (gf[1] * s + f * A * ω * cos(ω * x) * sin(ω * y),
     gf[2] * s + f * A * ω * sin(ω * x) * cos(ω * y))
end;

#%% code id=analytic_gradient_check
# The analytic gradients used above must agree with central finite differences.
# This is the job `numgrad` keeps: a check, not part of the descent itself.
let pts = [(0.0, 0.0), (2.0, -1.0), (-1.0, 1.5), (0.3, 0.7), (-1.2, 2.0)]
    maxdiff(f, ∇f) = maximum(maximum(abs.(collect(∇f(p...)) .- collect(numgrad(f, p...)))) for p in pts)
    g = (x, y) -> banana_sine(x, y; A = 0.5, ω = π)
    ∇g = (x, y) -> ∇banana_sine(x, y; A = 0.5, ω = π)
    errs = (F = maxdiff(F, ∇F), banana = maxdiff(banana, ∇banana), banana_sine = maxdiff(g, ∇g))
    # relative scale: the banana's gradient is large, so compare to its own size
    ok = errs.F < 1e-6 && errs.banana < 1e-3 && errs.banana_sine < 1e-3
    ok ? "✅ analytic gradients match finite differences: $errs" : "❌ mismatch: $errs"
end

#%% code id=local_minima_controls hidecode
@bind bs_A Slider(0.0:0.05:0.95; default=0.9, label="ripple amplitude A")
@bind bs_periods Slider(0.5:0.25:2.5; default=1.0, label="ripple periods across the window")
@bind bs_x0 Slider(-2.0:0.1:2.0; default=-1.2, label="start x (initial guess)")
@bind bs_y0 Slider(-1.0:0.1:3.0; default=1.0, label="start y")
@bind bs_lr Slider(0.001:0.001:0.01; default=0.003, label="learning rate (step size) η")
@bind bs_steps Slider(500:500:5000; default=3000, label="iterations")

#%% code id=local_minima_plot controls=[bs_A,bs_periods],[bs_x0,bs_y0],[bs_lr,bs_steps] hidecode
bs_ω = π * bs_periods / 2          # the window is 4 wide, so ω = 2π·(periods)/4
bs_f  = (x, y) -> banana_sine(x, y; A=bs_A, ω=bs_ω)
bs_∇f = (x, y) -> ∇banana_sine(x, y; A=bs_A, ω=bs_ω)
bs_path = gradient_descent(bs_∇f, (bs_x0, bs_y0), bs_lr, bs_steps)
bs_vals = [clamp(bs_f(p...), 1e-16, 1e12) for p in bs_path]
bs_idx = 1:max(1, length(bs_path) ÷ 400):length(bs_path)

bs_x = range(-2, 2; length=200)
bs_y = range(-1, 3; length=200)
BSZ = [log10(1 + bs_f(x, y)) for x in bs_x, y in bs_y]      # BSZ[x_index, y_index]

fig5c = Figure(size=(960, 440))
ax_c1 = Axis(fig5c[1, 1]; title="Banana × sine ripple (A = $(bs_A), $(bs_periods) period(s))", xlabel="x", ylabel="y")
contour!(ax_c1, bs_x, bs_y, BSZ; levels=24, linewidth=1.0)
lines!(ax_c1, first.(bs_path[bs_idx]), last.(bs_path[bs_idx]); color=:orange, linewidth=1.5)
scatter!(ax_c1, [last(bs_path)[1]], [last(bs_path)[2]]; color=:orange, markersize=10)
scatter!(ax_c1, [bs_x0], [bs_y0]; color=:lime, markersize=14)
scatter!(ax_c1, [1.0], [1.0]; color=:white, marker=:xcross, markersize=16)
xlims!(ax_c1, -2, 2); ylims!(ax_c1, -1, 3)

ax_c2 = Axis(fig5c[1, 2]; title="g at each iteration (log scale)", xlabel="iteration k", ylabel="g", yscale=log10)
lines!(ax_c2, (0:length(bs_vals)-1)[bs_idx], bs_vals[bs_idx]; color=:cyan, linewidth=2)
fig5c

#%% md id=basins
@md raw"""
<a id="sec-basins"></a>

### Basins of attraction

A single run only tells you where *that* start ends up. To see the whole picture, the **[basin of attraction](#dict-basin)** of a minimum is the set of starting points from which gradient descent ends in it. Every trap has its own basin, and the basins together tile the plane (apart from the thin ridges between them).

The plot below starts gradient descent from a \(13\times 13\) grid of points and runs each one **until it converges** (until \(\|\nabla g\|\le 10^{-3}\), the *stopping tolerance*). Then it groups the end points: runs that stop at the same minimum share one **color**, and the star (★) marks the minimum itself. The legend gives each minimum's location, its objective value \(g\), and how many of the starts fall into its basin. A gray dot means a run that did not converge within its iteration budget.

How to read it:

- **Each colored region is a basin.** The boundary between two regions is a ridge: start on one side and you slide into one trap, on the other side and you slide into a neighbor.
- **The global minimizer's basin is not automatically the biggest.** Compare the counts in the legend. A deep minimum can have a small basin, and a shallow trap a large one. Where you start matters as much as how good the minimum is.
- **Basins depend on the step size.** With a large fixed \(\eta\) a run can jump over a ridge, so the boundaries are no longer the smooth curves that the continuous downhill flow would give. Change \(\eta\) and the amplitude \(A\) above and watch the map change.
- **Why this matters for neural networks.** A network's loss is nonconvex too, with far more parameters. Its [initialization](#dict-initialization) (the random seed) picks the starting point, and so picks the basin. That is why [Experiment E](#sec-experiments) asks you to change only the seed.

The valley is slow: runs that start in the global minimizer's basin may need around 10 000 iterations, far more than the single-run plot above uses.
"""

#%% code id=basins_of_attraction hidecode
# Run gradient descent from a grid of starting points until each run reaches a stationary point
# (‖∇g‖ ≤ 1e-3), then group the endpoints into distinct minima. The valley is slow, so allow many iterations.
map_budget = 20_000
map_xs = range(-1.8, 1.8; length=13)
map_ys = range(-0.8, 2.8; length=13)
map_starts = vec([(x, y) for x in map_xs, y in map_ys])
map_ends = [gradient_descent(bs_∇f, s, bs_lr, map_budget; tol=1e-3)[end] for s in map_starts]

map_finite(e) = isfinite(e[1]) && isfinite(e[2]) && abs(e[1]) < 1e3 && abs(e[2]) < 1e3
map_conv = [map_finite(e) && hypot(bs_∇f(e...)...) <= 1e-3 for e in map_ends]

# Group converged endpoints that lie within 0.1 of each other; order the groups by objective value.
map_centers = Tuple{Float64,Float64}[]
for (i, e) in enumerate(map_ends)
    map_conv[i] || continue
    any(c -> hypot(c[1] - e[1], c[2] - e[2]) < 0.1, map_centers) || push!(map_centers, e)
end
sort!(map_centers; by = c -> bs_f(c...))
map_label = [map_conv[i] ? findfirst(c -> hypot(c[1] - e[1], c[2] - e[2]) < 0.1, map_centers) : 0
             for (i, e) in enumerate(map_ends)]

# One distinct color per minimum, chosen to differ from each other, from the dark background and from gray.
map_pal = Makie.Colors.distinguishable_colors(max(length(map_centers), 2),
    [Makie.Colors.RGB(0.11, 0.11, 0.11), Makie.Colors.RGB(0.5, 0.5, 0.5)]; dropseed=true)
map_col(k) = map_pal[k]
map_in(c) = -2 <= c[1] <= 2 && -1 <= c[2] <= 3
map_is_global(c) = hypot(c[1] - 1, c[2] - 1) < 0.1
map_name(k, c) = "$k  " * (map_is_global(c) ? "global" : "trap") *
    " ($(round(c[1]; digits=2)), $(round(c[2]; digits=2))), g=$(round(bs_f(c...); digits=2)), $(count(==(k), map_label)) starts" *
    (map_in(c) ? "" : "  [outside window]")

fig5d = Figure(size=(960, 520))
n_open = count(==(0), map_label)
ax_d = Axis(fig5d[1, 1]; xlabel="start x", ylabel="start y",
    title="$(length(map_centers)) local minima found ($(count(map_is_global, map_centers)) global)" *
          (n_open == 0 ? "; every run converged" : "; $n_open of $(length(map_label)) runs did not converge in $(map_budget) iterations"))
contour!(ax_d, bs_x, bs_y, BSZ; levels=24, linewidth=0.7, color=(:white, 0.25))

open_runs = findall(==(0), map_label)
isempty(open_runs) || scatter!(ax_d, first.(map_starts[open_runs]), last.(map_starts[open_runs]);
    color=(:gray, 0.6), markersize=9, label="not converged")

for k in eachindex(map_centers)
    members = findall(==(k), map_label)
    scatter!(ax_d, first.(map_starts[members]), last.(map_starts[members]); color=map_col(k), markersize=14)
end
for (k, c) in enumerate(map_centers)
    if map_in(c)       # a minimum inside the window: a star on the plot
        scatter!(ax_d, [c[1]], [c[2]]; color=map_col(k), marker=:star5, markersize=26,
            strokecolor=:white, strokewidth=2, label=map_name(k, c))
    else               # a minimum outside the window: legend entry only
        scatter!(ax_d, [NaN], [NaN]; color=map_col(k), marker=:star5, markersize=14, label=map_name(k, c))
    end
end
xlims!(ax_d, -2, 2); ylims!(ax_d, -1, 3)
Legend(fig5d[1, 2], ax_d, "minima (★), colored by basin"; labelsize=11, titlesize=12, patchsize=(14, 14))
fig5d

#%% md id=basins_exercise
@md raw"""
**Exercise: local minima and basins.**

1. Set \(A=0\). How many minima are there? What does the size of the basin tell you?
2. Raise \(A\). How does the number of minima change, and how does the size of the global minimizer's basin change? What changes when you go from one period to two?
3. In [stochastic gradient descent](#sec-sgd) we will add noise to the gradient. Why might that help escape a trap? Why might it not help in a deep one?
"""

#%% md id=sgd
@md raw"""
<a id="sec-sgd"></a>

## Many examples: stochastic gradient descent (stochastic approximation)

In ML the **[loss](#dict-loss)** (*objective*) is an average over \(N\) training examples, a **[finite-sum](#dict-finite-sum)** objective (the *empirical risk*):

\[
L(\theta)=\frac1N\sum_{i=1}^{N}\ell_i(\theta),
\qquad
\nabla L(\theta)=\frac1N\sum_{i=1}^{N}\nabla\ell_i(\theta).
\]

Using all \(N\) terms every step is the **[full-batch](#dict-full-batch)** method (the *deterministic* gradient method). **[Stochastic gradient descent (SGD)](#dict-sgd)** instead draws a random **mini-batch** \(\mathcal B\) and uses the *sampled gradient*

\[
g_{\mathcal B}(\theta)=\frac1{|\mathcal B|}\sum_{i\in\mathcal B}\nabla\ell_i(\theta).
\]

It is an *unbiased estimator* of \(\nabla L\): on average it points the right way, so we still go downhill along a jittery path, at a cost per step of \(|\mathcal B|\) instead of \(N\).

The simplest example is a **linear regression** (a *least-squares problem*): fit a line \(\hat y=mx+c\) to noisy points with the squared error \(\ell_i=(mx_i+c-y_i)^2\). The two **[parameters](#dict-parameters)** (*unknowns*) are \((m,c)\), and the contours are the *level sets* of \(L(m,c)\).

Try batch size 1 (pure SGD) against 40 (full batch, i.e. ordinary [gradient descent](#sec-descent)). Change the seed to get a different noise sequence.

**Exercise: the noise ball.** With a fixed step size, SGD never settles exactly on the minimizer: it wanders in a "noise ball" around it. Why? What happens to the ball when \(\eta\) shrinks? A **decreasing step-size rule** (the Robbins–Monro conditions, \(\sum\eta_k=\infty,\ \sum\eta_k^2<\infty\)) makes SGD converge. What does it cost you?
"""

#%% code id=sgd_data hidecode
# Noisy line: y = 1.5x - 0.5 + noise
lx, ly = let rng = Random.MersenneTwister(7)
    x = collect(range(-2, 2; length=40))
    x, 1.5 .* x .- 0.5 .+ 0.7 .* randn(rng, 40)
end

L(m, c) = sum(abs2, m .* lx .+ c .- ly) / length(lx)

# Per-batch gradient of the mean squared error
function batch_grad(m, c, idx)
    r = m .* lx[idx] .+ c .- ly[idx]
    (2 * sum(r .* lx[idx]) / length(idx), 2 * sum(r) / length(idx))
end

mvals = range(-1, 3; length=80)
cvals = range(-2.5, 1.5; length=80)
LZ = [L(m, c) for m in mvals, c in cvals];   # LZ[m_index, c_index]

#%% code id=sgd_method hidecode
# SGD: each step uses the gradient of a random mini-batch of `B` examples.
function run_sgd(θ0, η, n, B, seed)
    rng = Random.MersenneTwister(seed)
    m, c = θ0
    path = [(m, c)]
    for _ in 1:n
        idx = Random.randperm(rng, length(lx))[1:B]    # random mini-batch
        gm, gc = batch_grad(m, c, idx)
        m -= η * gm;  c -= η * gc
        push!(path, (m, c))
    end
    path
end;

#%% code id=sgd_controls hidecode
@bind sgd_batch Slider(1:40; default=1, label="batch size (samples per gradient)")
@bind sgd_lr Slider(0.005:0.005:0.15; default=0.03, label="learning rate (step size) η")
@bind sgd_steps Slider(10:10:300; default=100, label="iterations")
@bind sgd_seed Slider(1:20; default=1, label="seed (random draw)")

#%% code id=sgd_plot controls=[sgd_batch,sgd_lr],[sgd_steps,sgd_seed] hidecode
sgd_start = (-0.8, 1.0)
sgd_path = run_sgd(sgd_start, sgd_lr, sgd_steps, sgd_batch, sgd_seed)
full_path = run_sgd(sgd_start, sgd_lr, sgd_steps, length(lx), 1)   # full batch

fig6 = Figure(size=(960, 440))
ax8 = Axis(fig6[1, 1]; title="Parameter space (m, c)", xlabel="slope m", ylabel="intercept c")
contour!(ax8, mvals, cvals, LZ; levels=16, linewidth=1.0)
lines!(ax8, first.(full_path), last.(full_path); color=:cyan, linewidth=2, label="full-batch GD")
lines!(ax8, first.(sgd_path), last.(sgd_path); color=:orange, linewidth=1.3, label="SGD (batch $(sgd_batch))")
scatter!(ax8, [sgd_start[1]], [sgd_start[2]]; color=:lime, markersize=14)
scatter!(ax8, [1.5], [-0.5]; color=:white, marker=:xcross, markersize=14)
axislegend(ax8; position=:rb)

ax9 = Axis(fig6[1, 2]; title="The data and the final SGD line", xlabel="x", ylabel="y")
scatter!(ax9, lx, ly; color=:dodgerblue, markersize=9)
mf, cf = sgd_path[end]
lines!(ax9, [-2, 2], [mf*(-2)+cf, mf*2+cf]; color=:orange, linewidth=3)
fig6

#%% md id=neuron
@md raw"""
<a id="sec-neuron"></a>

## One neuron and a loss we can minimize

Back to [classification](#sec-classifier). The hard rule "\(s\ge 0\)" and its [0–1 loss](#dict-zero-one-loss) have no useful gradient. We soften the rule with the **[sigmoid](#dict-sigmoid)** (the *logistic function*) \(\sigma(z)=1/(1+e^{-z})\), which squeezes any number into \((0,1)\) and can be read as a probability:

\[
z=w_1x_1+w_2x_2+b,\qquad a=\sigma(z).
\]

This is a **[neuron](#dict-neuron)** (equivalently **logistic regression**: a linear model followed by a sigmoid). To measure how wrong it is on an example with label \(y\in\{0,1\}\), use the **[cross-entropy](#dict-cross-entropy)** loss (the *negative log-likelihood* of a Bernoulli model)

\[
\ell=-\big[y\ln a+(1-y)\ln(1-a)\big].
\]

It is small when \(a\) is close to \(y\) and huge when the neuron is confidently wrong: a smooth *surrogate* for the 0–1 loss. Averaged over the data it is our **objective** \(L\). Here every iteration uses all the data, so one **[epoch](#dict-epoch)** (one *pass over the data*) equals one iteration.

**The chain rule in action.** The loss depends on \(w\) only through a *composition* \(w\to z\to a\to\ell\), so we multiply the local derivatives:

\[
\frac{\partial \ell}{\partial w_j}
=\frac{\partial \ell}{\partial a}\,\frac{\partial a}{\partial z}\,\frac{\partial z}{\partial w_j}.
\]

We know \(\sigma'(z)=a(1-a)\) and \(\partial z/\partial w_j=x_j\). Everything hinges on \(\delta=\partial\ell/\partial z\), the **[error signal](#dict-error-signal)** (the *sensitivity* of the loss to \(z\)).

**Exercise: the error signal.** Compute \(\partial\ell/\partial a\), multiply by \(a(1-a)\), and simplify. You should get a strikingly simple result. Enter it in the next cell.
"""

#%% code id=neuron_exercise
# ✏️ Exercise: the error signal — δ = ∂ℓ/∂z as a function of a and y.
# (σ and ce — the sigmoid and the cross-entropy — are defined in the `shared_helpers` cell above.)
δ_mine(a, y) = 0.0   # TODO

#%% code id=neuron_exercise_check hidecode
h = 1e-6
cases = [(z, y) for z in (-2.0, -0.3, 0.8, 3.0) for y in (0, 1)]
errs4 = [abs(δ_mine(σ(z), y) - (ce(σ(z + h), y) - ce(σ(z - h), y)) / 2h) for (z, y) in cases]

if all(errs4 .< 1e-5)
    "✅ Correct: δ = ∂ℓ/∂z matches finite differences for both labels."
else
    "❌ Not yet (max error $(round(maximum(errs4); digits=4))). Try y = 1 and y = 0 separately, then combine."
end

#%% code id=neuron_defs
Xn = vcat(x1', x2')            # 2×N matrix of inputs
yn = Float64.(labels)

# One neuron trained by full-batch gradient descent, using δ = a − y
function train_neuron(X, y, η, epochs)
    w = [0.0, 0.0]; b = 0.0
    hist = [(copy(w), b)]
    losses = Float64[]
    for _ in 1:epochs
        a = σ.(X' * w .+ b)
        push!(losses, sum(ce.(clamp.(a, 1e-12, 1-1e-12), y)) / length(y))
        δ = a .- y
        w = w .- η .* (X * δ) ./ length(y)     # ∂L/∂w = mean(δ x)
        b = b - η * sum(δ) / length(y)         # ∂L/∂b = mean(δ)
        push!(hist, (copy(w), b))
    end
    hist, losses
end;

#%% code id=neuron_controls hidecode
@bind n_lr Slider(0.01:0.01:2.0; default=0.3, label="learning rate (step size) η")
@bind n_epoch Slider(0:200; default=0, label="epoch (iteration) to display")

#%% code id=neuron_train hidecode
# Trains once per learning rate; scrubbing the epoch slider does NOT retrain.
n_hist, n_losses = train_neuron(Xn, yn, n_lr, 200);

#%% code id=neuron_plot controls=n_lr,n_epoch hidecode
nw, nb = n_hist[n_epoch + 1]

gx = range(-2, 2; length=120)
# Makie expects P[x1_index, x2_index]
P = [σ(nw[1]*x + nw[2]*y + nb) for x in gx, y in gx]

fig7 = Figure(size=(960, 440))
ax10 = Axis(fig7[1, 1]; title="Neuron output at epoch $(n_epoch)", xlabel="x₁", ylabel="x₂", aspect=DataAspect())
hm = heatmap!(ax10, gx, gx, P; colormap=:balance, colorrange=(0, 1))
contour!(ax10, gx, gx, P; levels=[0.5], color=:white, linewidth=2)
scatter!(ax10, x1[labels .== 0], x2[labels .== 0]; color=:dodgerblue, strokecolor=:white, strokewidth=1.5, markersize=13)
scatter!(ax10, x1[labels .== 1], x2[labels .== 1]; color=:tomato, strokecolor=:white, strokewidth=1.5, markersize=13)
Colorbar(fig7[1, 2], hm; label="a = σ(z)")

ax11 = Axis(fig7[1, 3]; title="Loss", xlabel="epoch", ylabel="mean cross-entropy")
lines!(ax11, 1:length(n_losses), n_losses; color=:cyan, linewidth=2)
vlines!(ax11, [max(n_epoch, 1)]; color=:orange, linewidth=2)
fig7

#%% md id=backpropagation
@md raw"""
<a id="sec-backpropagation"></a>

## Two layers: forward pass, then backpropagation (reverse-mode differentiation)

One neuron can only draw a straight line (a *linear classifier*). Feed the outputs of several neurons into another and the boundary can bend. Our two-layer network has \(H\) neurons in a **[hidden layer](#dict-hidden-layer)** (a set of *intermediate variables*), so the loss is a **composition of functions**:

\[
z^{(1)}=W^{(1)}x+b^{(1)},\quad
h=\varphi\big(z^{(1)}\big),\quad
z^{(2)}=W^{(2)}h+b^{(2)},\quad
a=\sigma\big(z^{(2)}\big),\quad
\ell=-\big[y\ln a+(1-y)\ln(1-a)\big].
\]

Here \(x\in\mathbb R^2\), \(W^{(1)}\in\mathbb R^{H\times 2}\), \(W^{(2)}\in\mathbb R^{1\times H}\), and \(\varphi\) is the **activation function** (an elementwise *nonlinearity*, such as tanh or ReLU). Without a nonlinear \(\varphi\) the two layers would collapse into one linear map, and the network could not bend the boundary at all.

**[Backpropagation](#dict-backpropagation) is the chain rule applied from the output backwards**, reusing each result for the layer before it. In numerical-analysis language it is **reverse-mode automatic differentiation** (an *adjoint method*): one backward sweep yields the gradient with respect to *all* parameters at a cost comparable to one forward evaluation. Finite differences need an extra evaluation *per parameter*.

Define the **[error signal](#dict-error-signal)** (the *adjoint* of \(z^{(k)}\)) \(\delta^{(k)}=\partial\ell/\partial z^{(k)}\).

1. Output layer (the [error-signal exercise](#sec-neuron)): \(\delta^{(2)}=a-y\).
2. Gradients of the output parameters: \(\ \partial\ell/\partial W^{(2)}=\delta^{(2)}\,h^\top,\ \ \partial\ell/\partial b^{(2)}=\delta^{(2)}\).
3. Push the error back through \(W^{(2)}\) and the activation:
\[
\delta^{(1)}=\big(W^{(2)\top}\delta^{(2)}\big)\odot\varphi'\big(z^{(1)}\big).
\]
4. Gradients of the first layer: \(\ \partial\ell/\partial W^{(1)}=\delta^{(1)}\,x^\top,\ \ \partial\ell/\partial b^{(1)}=\delta^{(1)}\).

For a batch of \(N\) examples (columns of \(X\)), average these over the batch; in matrix form the outer products \(\delta h^\top\) and \(\delta x^\top\) become \(\Delta\,H^\top\) and \(\Delta\,X^\top\).

First the **forward pass** (the *function evaluation*): it computes the prediction and stores every intermediate value, because the backward pass needs them. **Exercise: backpropagation** is to write the backward pass yourself; the **[gradient check](#dict-gradient-check)** (a *finite-difference derivative test*) tells you whether you got it right.
"""

#%% code id=network_forward hidecode
# Activation functions together with their derivatives φ'(z)
ACT = Dict(
    :tanh    => (f = tanh,               df = z -> 1 - tanh(z)^2),
    :relu    => (f = z -> max(0.0, z),   df = z -> z > 0 ? 1.0 : 0.0),
    :sigmoid => (f = σ,                  df = z -> σ(z) * (1 - σ(z))),
)

# Parameters: W1 (H×2), b1 (H), W2 (1×H), b2 (1). Small random start.
function init_params(H; seed=1, scale=0.5)
    rng = Random.MersenneTwister(seed)
    (W1 = scale .* randn(rng, H, 2), b1 = zeros(H),
     W2 = scale .* randn(rng, 1, H), b2 = zeros(1))
end

# Forward pass for a batch X (2×N). Keep intermediates for backprop.
function forward(p, X, act)
    Z1 = p.W1 * X .+ p.b1          # H×N
    H  = act.f.(Z1)
    Z2 = p.W2 * H .+ p.b2          # 1×N
    A  = σ.(Z2)
    (; Z1, H, Z2, A)
end

# Mean cross-entropy over the batch
function loss(p, X, y, act)
    A = clamp.(vec(forward(p, X, act).A), 1e-12, 1 - 1e-12)
    Statistics.mean(ce.(A, y))
end;

#%% code id=backpropagation_exercise
# ✏️ Exercise: backpropagation — write the backward pass yourself.
# Return the gradient of the MEAN loss with respect to each parameter.
function backward_mine(p, X, y, act)
    N = size(X, 2)
    c = forward(p, X, act)                  # c.Z1, c.H, c.Z2, c.A are available
    Δ2 = reshape(vec(c.A) .- y, 1, N)       # δ⁽²⁾ = a - y   (1×N)

    dW2 = zeros(size(p.W2))   # TODO: (1/N) Δ2 Hᵀ
    db2 = zeros(size(p.b2))   # TODO
    Δ1  = zeros(size(c.Z1))   # TODO: (W2ᵀ Δ2) ⊙ φ'(Z1)
    dW1 = zeros(size(p.W1))   # TODO: (1/N) Δ1 Xᵀ
    db1 = zeros(size(p.b1))   # TODO

    (W1 = dW1, b1 = db1, W2 = dW2, b2 = db2)
end;

#%% code id=grad_check
# Gradient check: compare an analytic gradient with central finite differences.
function numeric_grad(p, X, y, act; h=1e-6)
    map(p) do arr
        g = zeros(size(arr))
        for i in eachindex(arr)
            old = arr[i]
            arr[i] = old + h; lp = loss(p, X, y, act)
            arr[i] = old - h; lm = loss(p, X, y, act)
            arr[i] = old
            g[i] = (lp - lm) / 2h
        end
        g
    end
end

# Largest absolute difference over all parameters
function grad_error(backward_fn, p, X, y, act)
    ga = backward_fn(p, X, y, act)
    gn = numeric_grad(p, X, y, act)
    maximum(maximum(abs.(ga[k] .- gn[k])) for k in keys(p))
end;

#%% code id=backpropagation_exercise_check hidecode
let p = init_params(4; seed=3), err = grad_error(backward_mine, p, Xn, yn, ACT[:tanh])
    if err < 1e-6
        "✅ Backpropagation verified: max |analytic − numeric| = $(round(err; sigdigits=2))."
    else
        "❌ Not yet: max |analytic − numeric| = $(round(err; sigdigits=3)). Fill in the TODO lines one at a time; the output layer is easiest."
    end
end

#%% code id=backward_solution collapsed
# 🔑 Reference solution for the backpropagation exercise (try it yourself first!).
# The rest of the notebook uses this version, so it works even if yours isn't done.
function backward(p, X, y, act)
    N = size(X, 2)
    c = forward(p, X, act)
    Δ2 = reshape(vec(c.A) .- y, 1, N)                 # δ⁽²⁾ = a − y
    dW2 = Δ2 * c.H' ./ N
    db2 = [sum(Δ2) / N]
    Δ1  = (p.W2' * Δ2) .* act.df.(c.Z1)              # δ⁽¹⁾ = (W2ᵀ δ⁽²⁾) ⊙ φ'(z⁽¹⁾)
    dW1 = Δ1 * X' ./ N
    db1 = vec(sum(Δ1; dims=2)) ./ N
    (W1 = dW1, b1 = db1, W2 = dW2, b2 = db2)
end

# Verify against finite differences for every activation
[(name, grad_error(backward, init_params(5; seed=2), Xn, yn, ACT[name])) for name in (:tanh, :relu, :sigmoid)]

#%% md id=training
@md raw"""
<a id="sec-training"></a>

## Training the network on 2-D datasets (solving the optimization problem)

Now we put everything together. In each **[epoch](#dict-epoch)** (one *pass over the data*) we shuffle the training set, cut it into mini-batches, and for each mini-batch we run

1. a **forward pass** (evaluate the objective),
2. the **backward pass** (the gradient, via [backpropagation](#sec-backpropagation)),
3. the **update** (the *step*) \(\theta\leftarrow\theta-\eta\,g_{\mathcal B}\).

This is exactly [stochastic gradient descent](#sec-sgd), applied to the cross-entropy loss of the two-layer network.

To *see* what the network learned we evaluate it on a fine grid and color each grid point by its output \(a\in(0,1)\). The white curve is the **[decision boundary](#dict-decision-boundary)** (the \(a=0.5\) *level set*). The training points are drawn on top; a white outline marks a misclassified point.

We use five datasets of increasing difficulty:

| dataset | what it tests |
|---|---|
| **linear** | two blobs a straight line can separate; one neuron would do, and its loss is *convex* in \((w,b)\) |
| **XOR** | opposite corners share a class: no single line works |
| **circles** | one class surrounded by the other |
| **spiral** | two interleaved arms (1.5 turns): needs a flexible boundary and patient training |
| **swiss roll** | the same idea with 2 turns: arms are packed tighter, so it needs a wider network and more epochs |

With a hidden layer the loss is **[nonconvex](#dict-nonconvex)** in the parameters: it has *local minima* and *saddle points*, so different **[initializations](#dict-initialization)** (*starting points*) can end in different solutions, just like the different [basins of attraction](#sec-basins) above. Start with *linear*, then move to *XOR* and watch what the hidden layer buys us. The [experiments](#sec-experiments) that follow give you a checklist.
"""

#%% code id=datasets collapsed
DATASETS = [:linear => "Linearly separable", :xor => "XOR", :circles => "Circles",
            :spiral => "Spiral", :swissroll => "Swiss roll (2 interleaved spirals)"]

# Returns X (2×N) and labels y (length N, values 0/1)
function make_dataset(kind::Symbol; n=200, noise=0.25, seed=0)
    rng = Random.MersenneTwister(seed)
    m = n ÷ 2
    if kind == :linear
        A = [-1.0, -0.8] .+ noise .* randn(rng, 2, m)
        B = [ 1.0,  0.8] .+ noise .* randn(rng, 2, m)
    elseif kind == :xor
        k = n ÷ 4
        pts(cx, cy) = [cx, cy] .+ noise .* randn(rng, 2, k)
        A = hcat(pts(1, 1),  pts(-1, -1))     # class 0: x₁x₂ > 0
        B = hcat(pts(-1, 1), pts(1, -1))      # class 1: x₁x₂ < 0
    elseif kind == :circles
        θA = 2π .* rand(rng, m);  θB = 2π .* rand(rng, m)
        A = 0.7 .* [cos.(θA)'; sin.(θA)'] .+ noise .* 0.5 .* randn(rng, 2, m)
        B = 1.6 .* [cos.(θB)'; sin.(θB)'] .+ noise .* 0.5 .* randn(rng, 2, m)
    elseif kind == :spiral || kind == :swissroll
        # Two interleaved arms, one per class: radius grows with t, the angle winds with t.
        # `turns` sets how many times each arm winds around the origin (more turns = harder).
        turns = kind == :spiral ? 1.5 : 2.0
        t = range(0.12, 1.0; length=m)
        arm(φ) = hcat([(2.0t[i]) .* [cos(2π*turns*t[i] + φ), sin(2π*turns*t[i] + φ)] for i in 1:m]...) .+ noise .* 0.4 .* randn(rng, 2, m)
        A = arm(0.0); B = arm(π)
    else
        error("unknown dataset $kind")
    end
    hcat(A, B), vcat(zeros(size(A, 2)), ones(size(B, 2)))
end;

#%% code id=training_loop hidecode
copy_params(p) = map(copy, p)

accuracy(p, X, y, act) = Statistics.mean((vec(forward(p, X, act).A) .>= 0.5) .== (y .== 1))

# Mini-batch SGD with hand-made backprop.
# Returns per-epoch losses/accuracies and parameter snapshots (to scrub through training).
function train!(p, X, y, act; η=0.1, batch=16, epochs=300, seed=1, nsnap=50)
    rng = Random.MersenneTwister(seed)
    N = size(X, 2)
    losses = zeros(epochs); accs = zeros(epochs)
    snap_at = unique(round.(Int, range(1, epochs; length=nsnap)))
    snaps = Any[copy_params(p)]                       # snapshot 1 = initial parameters
    for e in 1:epochs
        perm = Random.randperm(rng, N)
        for s in 1:batch:N
            idx = perm[s:min(s + batch - 1, N)]
            g = backward(p, X[:, idx], y[idx], act)
            for k in keys(p)
                p[k] .-= η .* g[k]                    # θ ← θ − η g
            end
        end
        losses[e] = loss(p, X, y, act)
        accs[e] = accuracy(p, X, y, act)
        e in snap_at && push!(snaps, copy_params(p))
    end
    (; losses, accs, snaps, snap_at)
end;

#%% code id=training_controls collapsed hidecode
@bind ds_kind Select(DATASETS; label="dataset")
@bind net_H Slider(1:32; default=4, label="hidden neurons H")
@bind net_act Select([:tanh => "tanh", :relu => "ReLU", :sigmoid => "sigmoid"]; label="activation (nonlinearity)")
@bind net_lr Slider(0.005:0.005:1.0; default=0.1, label="learning rate (step size) η")
@bind net_batch Select([1 => "1 (pure SGD)", 4 => "4", 16 => "16", 64 => "64", 200 => "200 (full batch)"], 16; label="batch size")
@bind net_epochs Slider(50:50:2000; default=300, label="epochs (passes over data)")
@bind net_seed Slider(1:20; default=1, label="init seed (initial guess)")
@bind net_view Slider(0:50; default=50, label="training snapshot (scrub)")

#%% code id=training_run hidecode
# Select controls give a Choice; `.value` is the underlying Symbol/Int.
ds = ds_kind.value
Xd, yd = make_dataset(ds; seed=42)
act = ACT[net_act.value]
params0 = init_params(net_H; seed=net_seed)
fit = train!(copy_params(params0), Xd, yd, act; η=net_lr, batch=net_batch.value, epochs=net_epochs, seed=net_seed)
(; dataset=ds, final_loss=round(fit.losses[end]; digits=4), final_accuracy=fit.accs[end])

#%% code id=training_plot controls=[ds_kind,net_H,net_act],[net_lr,net_batch,net_epochs],[net_seed,net_view] hidecode
snap_i = clamp(net_view + 1, 1, length(fit.snaps))
p_view = fit.snaps[snap_i]
epoch_view = snap_i == 1 ? 0 : fit.snap_at[snap_i - 1]

# Evaluate the network on a grid covering the plane
gr = range(-2.5, 2.5; length=140)
G = vcat(repeat(collect(gr)', length(gr)) |> vec |> permutedims,    # x₁ varies along columns
         repeat(collect(gr), length(gr)) |> permutedims)             # x₂ varies along rows
Agrid = reshape(vec(forward(p_view, G, act).A), length(gr), length(gr))   # [x₂ index, x₁ index]

pred = vec(forward(p_view, Xd, act).A) .>= 0.5
miss = pred .!= (yd .== 1)

fig9 = Figure(size=(1000, 440))
ax12 = Axis(fig9[1, 1]; title="$(ds_kind.label): epoch $(epoch_view), accuracy $(round(100*mean(.!miss); digits=1))%",
            xlabel="x₁", ylabel="x₂", aspect=DataAspect())
hm9 = heatmap!(ax12, gr, gr, Agrid'; colormap=:balance, colorrange=(0, 1))
contour!(ax12, gr, gr, Agrid'; levels=[0.5], color=:white, linewidth=2.5)
scatter!(ax12, Xd[1, yd .== 0], Xd[2, yd .== 0]; color=:dodgerblue, strokecolor=:black, strokewidth=1, markersize=10)
scatter!(ax12, Xd[1, yd .== 1], Xd[2, yd .== 1]; color=:tomato, strokecolor=:black, strokewidth=1, markersize=10)
any(miss) && scatter!(ax12, Xd[1, miss], Xd[2, miss]; color=:transparent, strokecolor=:white, strokewidth=2, markersize=18)
Colorbar(fig9[1, 2], hm9; label="network output a")

ax13 = Axis(fig9[1, 3]; title="Training loss", xlabel="epoch", ylabel="mean cross-entropy")
lines!(ax13, 1:length(fit.losses), fit.losses; color=:cyan, linewidth=2)
vlines!(ax13, [max(epoch_view, 1)]; color=:orange, linewidth=2)
fig9

#%% md id=experiments
@md raw"""
<a id="sec-experiments"></a>

## Experiments

Use the controls beside the figure in [training](#sec-training). Write down what you observe, then explain it with the mathematics above. Unfamiliar term? See the [dictionary](#dictionary).

**A. Linear data.** Set \(H=1\). One hidden neuron is enough. How many [epochs](#dict-epoch) (passes over the data) until the accuracy reaches 100%? Scrub the *training snapshot* slider to watch the [decision boundary](#dict-decision-boundary) rotate and settle.

**B. XOR.** With \(H=1\) the accuracy gets stuck. Why can't one hidden neuron work? Try \(H=2\), then \(H=4\). What does each hidden neuron contribute geometrically?

**C. Circles.** Which activation function gives the smoothest boundary? How few hidden neurons suffice?

**D. Spiral.** With \(H=4\) the network fails. Raise \(H\) and the epochs, and tune the [learning rate](#dict-learning-rate) (step size) \(\eta\). Hint: tanh with \(H=16\) can reach 100%. Does the batch size (the noise in the sampled gradient, see [SGD](#dict-sgd)) change how fast the loss falls? The final result? Does ReLU need more neurons than tanh?

**E. Initialization and step size.** Fix XOR with \(H=3\) and change only the *init seed* (the [initial guess](#dict-initialization) \(\theta_0\)). Do all seeds succeed? What does that say about a [nonconvex](#dict-nonconvex) objective? Compare with the [basins of attraction](#sec-local-minima) map. Then raise \(\eta\) above 0.5 and connect what you see to the \(2/\lambda_{\max}\) rule in [gradient descent](#sec-descent).

**F. Swiss roll (the challenge).** Here more capacity alone is not enough. One setting that is known to work: tanh, \(H=32\), \(\eta=0.3\), batch 16, 2000 epochs (about 99% accuracy). Now change *only* \(\eta\) to 0.1 and to 0.5. Both do worse. Explain each failure: one is too slow, the other too noisy. What would a decreasing step-size rule (as in the [SGD](#sec-sgd) exercise) do?

**G. Your own dataset.** Unfold the cell `datasets` (folded above), add a dataset such as two moons or a checkerboard, include it in `DATASETS`, and find the network size it needs.
"""

#%% md id=dictionary
@md raw"""
<a id="dictionary"></a>

## Dictionary: machine learning ↔ numerical optimization

Every bold term in the text links here, and each row links back to the section that introduces it.

| machine learning | numerical optimization | where |
|---|---|---|
| <a id="dict-parameters"></a>parameters, weights | decision variables, unknowns | [classifier](#sec-classifier), [landscape](#sec-landscape) |
| <a id="dict-features"></a>features, inputs | independent variables (problem data) | [classifier](#sec-classifier) |
| <a id="dict-loss"></a>loss, cost function | objective function | [minimum](#sec-minimum) |
| <a id="dict-training"></a>training, fitting | solving the optimization problem | [minimum](#sec-minimum), [training](#sec-training) |
| <a id="dict-loss-landscape"></a>loss landscape | graph of the objective; level sets (contours) | [landscape](#sec-landscape) |
| <a id="dict-decision-boundary"></a>decision boundary | zero level set of the score | [classifier](#sec-classifier), [training](#sec-training) |
| <a id="dict-zero-one-loss"></a>0–1 loss (error count) | piecewise-constant objective, useless gradient | [classifier](#sec-classifier), [neuron](#sec-neuron) |
| <a id="dict-gradient-descent"></a>gradient descent | steepest descent method | [descent](#sec-descent) |
| <a id="dict-learning-rate"></a>learning rate | step size (step length) | [descent](#sec-descent) |
| <a id="dict-initialization"></a>initialization | initial guess, starting point \(\theta_0\) | [descent](#sec-descent), [local minima](#sec-local-minima), [training](#sec-training) |
| <a id="dict-training-curve"></a>training curve | convergence history | [descent](#sec-descent) |
| <a id="dict-vanishing-gradient"></a>vanishing gradient (at the optimum) | stationarity, \(\nabla F=0\) (first-order necessary condition) | [gradient](#sec-gradient) |
| <a id="dict-divergence"></a>learning rate too large, divergence | step size above \(2/\lambda_{\max}\), unstable iteration | [descent](#sec-descent), [banana](#sec-banana) |
| <a id="dict-ravines"></a>ravines, badly scaled loss | ill-conditioning: large condition number \(\lambda_{\max}/\lambda_{\min}\) | [banana](#sec-banana) |
| <a id="dict-momentum"></a>momentum | heavy-ball method | [banana](#sec-banana) |
| <a id="dict-local-minima"></a>bad local minima, traps | non-global local minimizers | [local minima](#sec-local-minima) |
| <a id="dict-basin"></a>basin of attraction | set of starting points that lead to one minimum | [basins](#sec-basins) |
| <a id="dict-full-batch"></a>full-batch gradient descent | deterministic gradient method | [descent](#sec-descent), [sgd](#sec-sgd) |
| <a id="dict-sgd"></a>SGD, mini-batch gradient | stochastic approximation, sampled gradient | [sgd](#sec-sgd) |
| <a id="dict-finite-sum"></a>finite-sum loss, empirical risk | sum of many terms; sample-average objective | [sgd](#sec-sgd) |
| <a id="dict-epoch"></a>epoch | one pass (sweep) over the data | [neuron](#sec-neuron), [training](#sec-training) |
| <a id="dict-sigmoid"></a>sigmoid | logistic function | [neuron](#sec-neuron) |
| <a id="dict-cross-entropy"></a>cross-entropy | negative log-likelihood | [neuron](#sec-neuron) |
| <a id="dict-neuron"></a>logistic regression, neuron | generalized linear model with logistic link | [neuron](#sec-neuron) |
| <a id="dict-hidden-layer"></a>hidden layer, activation function | intermediate variables, elementwise nonlinearity | [backpropagation](#sec-backpropagation) |
| <a id="dict-error-signal"></a>error signal \(\delta\) | adjoint (sensitivity) of the intermediate variable | [neuron](#sec-neuron), [backpropagation](#sec-backpropagation) |
| <a id="dict-backpropagation"></a>backpropagation | reverse-mode automatic differentiation (adjoint method) | [backpropagation](#sec-backpropagation) |
| <a id="dict-gradient-check"></a>gradient check | finite-difference derivative test | [gradient](#sec-gradient), [backpropagation](#sec-backpropagation) |
| <a id="dict-nonconvex"></a>nonconvex loss, local minima | nonconvex objective, local minimizers, saddle points | [training](#sec-training) |
"""

# ╔═╡ Slate.config · per-notebook settings (Settings panel)
#   docid = 104f421b-73d0-4ad9-831e-7b6ae450e3cb
#   format = 2
# ╚═╡
