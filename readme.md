# Brain Portfolio System 

**Build a long-only "brain portfolio"**: several small modules (experts) + one coordinator that blends them. The system adapts (plasticity) so if one part struggles, others compensate—like a healthy brain/community.

## Goal 
- Turn **$500k → ≥ $1.5M by 2036** (~11.6% CAGR) **and** fund **$10k/yr from 2029** onward (growing with AUM)

## Quick Start

```bash
# Install dependencies
pip install numpy pandas scikit-learn scipy

# Run the system
python brain_portfolio.py
```

## Architecture Overview

### Modules (the "brain regions")

1. **Prefrontal (Growth/AI)** – tech & AI growth
   - Model: Gradient Boosted Trees (XGBoost/LightGBM)
   - Signals: Multi-horizon momentum, earnings revisions, quality factors
   - Focus: Technology and communication sectors

2. **Hippocampus (Infra/REITs)** – real assets/infrastructure
   - Model: Bayesian Ridge Regression
   - Signals: Dividend yield, AFFO proxies, rate betas, inflation expectations
   - Focus: REITs and infrastructure assets

3. **Cerebellum (Income)** – Treasuries/IG/munis/cash to fund withdrawals
   - Model: Constrained optimizer (not ML)
   - Signals: Yield curve, OAS, credit quality, duration
   - Focus: Fixed income for withdrawal coverage

4. **Amygdala (Defensive)** – healthcare/staples/low-vol
   - Model: Logistic classifier for downside protection
   - Signals: Credit spreads, VIX, defensive momentum
   - Focus: Risk reduction during stress

5. **Plasticity Hub (Impact/Green)** – clean energy, efficiency, urban tech
   - Model: Small GRU or Gradient Boosted Trees
   - Signals: Clean-tech momentum, policy indicators, rate sensitivity
   - Focus: ESG and impact investments

6. **Construction Sleeve (Philadelphia)** – materials/energy/infra mix that **moves with local construction costs**
   - Model: Bayesian Ridge for PPI overlay
   - Signals: Material ETFs (XLB/XME/COPX), PPIs, energy curve
   - Target: **Correlation ≥ 0.6** with Philadelphia construction costs

## How it Works (in plain English)

- Each module has a tiny model that scores **return (alpha)** and **risk**
- Temperature-scaled softmax converts scores to weights: `w_i = exp(score_i/T) / Σ exp(score_j/T)`
- A **Coordinator** combines module signals into **final weights** (long-only, sum to 1)
- **Rebalance monthly**, making small tweaks most of the time; bigger shifts only in stress (plasticity)

## Mathematical Framework

### Scoring Function
```
score_i = α_i - κ·risk_i
```
Where:
- `α_i` = expected return for asset i
- `κ` = risk penalty parameter (default: 0.3)
- `risk_i` = volatility or downside risk measure

### Weight Calculation
```
w_i = exp(score_i/T) / Σ_j exp(score_j/T)
```
Where:
- `T` = temperature parameter (small → concentrated, large → diversified)

## Guardrails (non-negotiables)

- **Income floor** so $10k+/yr is always covered (ICR ≥ **1.2×** next 12 months)
- **Module caps** (min ~5%, max ~30%); **max single ETF 15%**
- **Volatility band** target ~8–12%
- **Turnover budget** to avoid churn (max 15% annually)

## Special Features

### Construction Readiness Metrics
- **Correlation** with Philly construction proxy (aim ≥ **+0.6**)
- **CRR** = sleeve NAV / inflation-adjusted 2036 budget (target **≥1.0 by 2034**, buffer 1.1–1.2 by 2036)

### Plasticity Mechanism
- Module weights adapt based on recent performance
- Momentum-based reweighting with bounds
- Ensures system resilience when modules underperform

## Testing & Validation

### 1. Backtesting
- Walk-forward splits only (no lookahead bias)
- Purged training sets
- Monthly rebalancing simulation

### 2. Stress Scenarios
- **Rate shock**: Treasury yields jump to 7%
- **Commodity spike**: +30% energy, +25% materials
- **Recession**: -25% equity, wide credit spreads
- **Tech crash**: -35% technology sector

### 3. Lesion Tests
- Disable each module individually
- Measure performance impact
- Verify system resilience

## Performance Targets (Scorecard)

| Metric | Target | Status |
|--------|--------|--------|
| CAGR | ~11.6% | ⏳ |
| Max Drawdown | < 20% | ⏳ |
| ICR | ≥ 1.2× | ✓ |
| Construction Correlation | ≥ 0.6 | ⏳ |
| Annual Turnover | < 15% | ✓ |

## Code Structure

```
brain_portfolio.py
├── Configuration
│   └── PortfolioConfig          # Global parameters
├── Modules
│   ├── BaseModule               # Abstract base class
│   ├── PrefrontalModule         # Growth/AI
│   ├── HippocampusModule        # Infrastructure
│   ├── CerebellumModule         # Income optimization
│   ├── AmygdalaModule           # Defensive
│   ├── PlasticityModule         # Green/Impact
│   └── ConstructionModule       # Philly tracking
├── Coordination
│   └── PortfolioCoordinator     # Blends module signals
├── Main System
│   └── BrainPortfolio           # Main portfolio manager
└── Testing
    ├── StressTestEngine         # Scenario analysis
    └── LesionTester             # Module resilience

```

## Model Choices (Minimal Configuration)

| Module | Primary Model | Why |
|--------|--------------|-----|
| Prefrontal | Gradient Boosted Trees | Handles nonlinearity, small data friendly, SHAP explainable |
| Hippocampus | Bayesian Ridge | Probabilistic, uncertainty-aware sizing |
| Cerebellum | Constrained Optimizer | Direct income optimization with guardrails |
| Amygdala | Logistic Classifier | Simple downside event prediction |
| Plasticity | Small GBM | Avoids overfitting on limited ESG data |
| Construction | Bayesian Ridge | Tracks PPI with uncertainty bands |

## 10-Week Implementation Plan

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1-2 | Data & Rules | Data pipeline, universe definition |
| 3-4 | Module Stubs | All 6 modules with basic signals |
| 5 | Guardrails | Income floor, caps, volatility bands |
| 6 | Construction | Philadelphia tracking sleeve |
| 7-8 | Backtests | Walk-forward validation, metrics |
| 9 | Interpretability | SHAP values, weight attribution |
| 10 | Documentation | Pitch deck, visuals, final report |

## Usage Example

```python
from brain_portfolio import BrainPortfolio, PortfolioConfig

# Initialize
config = PortfolioConfig(
    initial_capital=500_000,
    target_capital_2036=1_500_000,
    annual_withdrawal_base=10_000
)
portfolio = BrainPortfolio(config)

# Train on historical data
portfolio.train_all_modules(historical_data)

# Monthly rebalance
weights = portfolio.rebalance(current_market_data)

# Stress test
stress_results = stress_tester.run_all_scenarios(market_data)

# Check resilience
lesion_results = lesion_tester.test_all_lesions(market_data)
```

## Key Innovations

1. **Neuromorphic Design**: Mimics brain's modular, adaptive structure
2. **Construction Hedge**: Unique sleeve tracking local construction costs
3. **Plasticity**: Dynamic module weighting based on performance
4. **Multi-Model Ensemble**: Each module uses the simplest effective model
5. **Temperature Softmax**: Elegant diversification control

## Risk Management

- **Module Diversification**: No single module > 30%
- **Asset Caps**: No single ETF > 15%
- **Income Coverage**: Always maintain 1.2× coverage ratio
- **Volatility Bands**: Target 8-12% portfolio volatility
- **Turnover Control**: Limit trading costs

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scikit-learn`: Machine learning models
- `scipy`: Optimization routines

## Future Enhancements

- [ ] Real-time data integration
- [ ] Advanced correlation modeling
- [ ] Deep learning modules (TensorFlow/PyTorch)
- [ ] Interactive dashboard
- [ ] Monte Carlo scenario planning
- [ ] Tax-loss harvesting
- [ ] Alternative data sources

## Story for Connor (closing line)

> "A **plastic**, community-minded portfolio: when one sector struggles, the rest step up—**funding today's programs** and **protecting the 2036 build** against rising construction costs."

---

**Note**: This system is designed for educational and research purposes. Always consult with financial advisors for actual investment decisions.

## License

MIT License - See LICENSE file for details

## Contact

For questions or contributions, please open an issue or pull request.