# Copilot Instructions for quant-modelling

## Project Overview
A quantitative modeling codebase. This document will be updated as the architecture solidifies.

Use Pytorch for model implementation to facilitate autodifferentiation. Don't install the GPU version to save disk space.

Create a flexibile pattern where pricing is done using the triple of (Model, Security, Calculator). This allows to price a 
European option under Black model using Analytics, PDE or Monte-Carlo simulation.

## Directory Structure (Emerging)

- Analytics: Core numerical and statistical functions. Linear algebra utilities, optimizers and random number generation. Includes a Brownian bridge for Sobol numbers under Monte-Carlo
- Models: Implementation of quantitative models (e.g., Black-Scholes, Dupire local volatility, Heston stochastic volatility)
- MarketData: Market data structures such as spot, curves and volatility surfaces (SABR, SVI, SSVI)
- Calculators: Analytical, Monte-Carlo simulations, PDE solvers, and other numerical methods. All Monte-Carlo simulations should vectorize over the number of paths so models need to support this.
- Securities: Option payoff definitions and related financial instruments (European, American, Barrier options)
- Scenarios: Market scenario generation and management

### Dependencies & Tools
- **Core**: NumPy, Pandas for numerical computing
- **Modeling**: scikit-learn or similar for algorithmic work
- **Testing**: pytest (assumed standard)
- **Documentation**: docstrings with examples

## Key Patterns to Follow (As Established)

Create a Python virtual environment for development. Install all packages and build a setup.py for easy installation.
All coding should adhere to PEP 8 standards. Use type hints extensively for function signatures. 

### Model Development
When creating quantitative models:
- Separate model logic from data handling
- Use type hints for numerical parameters
- Document assumptions and mathematical formulations in docstrings
- Include unit tests alongside model implementations

### Data Handling
- Immutable data flows where possible
- Explicit timestamp/timezone handling for time series
- Validation of data shapes and types at entry points

### Testing & Validation
- Unit tests for mathematical functions and model calculations
- Integration tests for data pipelines
- Performance benchmarks for computationally intensive operations

## Future Guidance
As components are added, update this file to include:
1. Module interdependencies and data flows
2. Build/test commands specific to each component
3. Performance optimization patterns used in the codebase
4. External API/service integration patterns
5. Deployment and configuration conventions

## Getting Started
1. Check the README.md for project overview and setup instructions
2. Look at established modules for code style examples
3. Run tests locally: `pytest` (once test suite is created)

---

**Last Updated**: December 29, 2025
