# From functions to neural networks

An interactive [Kaimon Slate](https://github.com/kahliburke/KaimonSlate.jl) notebook (`ann-intro.jl`) that bridges machine learning and numerical optimization. Each idea is introduced as an ML term alongside its optimization counterpart (parameters = decision variables, loss = objective, training = solving the optimization problem), so you can read papers from either side. The notebook builds from a linear classifier through loss landscapes, gradients, gradient descent (including ill-conditioned valleys and local minima), and stochastic gradient descent. It then derives and implements backpropagation for a two-layer network in Julia and trains it on linearly separable data, XOR, circles and spirals. Sliders change the mathematics, not just the plots. There are checked exercises along the way and a two-way ML/optimization dictionary at the end.

## Installing Julia

The recommended way is [juliaup](https://github.com/JuliaLang/juliaup), the official Julia installer and version manager.

- **macOS / Linux:**
  ```
  curl -fsSL https://install.julialang.org | sh
  ```
- **Windows:**
  ```
  winget install --name Julia --id 9NJNWW8PVKMN -e -s msstore
  ```

Open a new terminal and check the install with `julia --version`. Other installers and platforms are listed at <https://julialang.org/downloads/>.

## Running the notebook

1. Clone the repository:
   ```
   git clone https://github.com/AUTH-CSAL/ann-intro.git
   cd ann-intro
   ```
2. Install Kaimon Slate as a Julia app (once). In the Julia REPL, press `]` to enter the package mode and run:
   ```
   pkg> app add KaimonSlate
   ```
   This installs the `slate` command into `~/.julia/bin`. Make sure that directory is on your `PATH`.
3. Exit Julia (`exit()` or Ctrl-D), then start the notebook from the shell:
   ```
   slate ann-intro.jl
   ```
   The notebook manages its own environment, so there is no need to add `KaimonSlate` or any other package to a project. See the [KaimonSlate documentation](https://github.com/kahliburke/KaimonSlate.jl) for details.
