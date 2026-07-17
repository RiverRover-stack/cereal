You are analyzing a competitor website to inform the design and positioning of
[Tool Name], an explainable sales forecasting tool for e-commerce sellers
(see design-doc.md for full context: problem, solution, MVP scope).

Competitor to analyze: Forthcast — https://www.forthcast.io/
(AI inventory/demand forecasting for Shopify, $19.99/month flat, SKU-level
forecasts up to 12 months, reorder alerts, "forecast accuracy engine" that
tracks prediction bias over time.)

Fetch and review these pages specifically:
- https://www.forthcast.io/ (homepage)
- https://www.forthcast.io/pricing
- https://www.forthcast.io/features
- https://www.forthcast.io/tools/reorder-point-calculator (their free-tool SEO hook)

Analyze and report back on:

1. **Information architecture** — how do they structure the homepage? What's
   the order of sections (hero, problem, features, social proof, pricing,
   FAQ)? What's above the fold?

2. **How they explain a technical/AI product to a non-technical buyer** —
   what language do they use to make "AI forecasting" feel trustworthy and
   concrete rather than vague? Pull specific phrasing patterns (not to copy
   verbatim, but to understand the technique).

3. **Free-tool-as-SEO-hook pattern** — how is their Reorder Point Calculator
   page structured? How does it funnel a visitor toward the paid product?
   What does the free tool's UI/UX look like, and how much value does it
   give away before asking for signup?

4. **Pricing page structure** — how do they present a single flat-rate plan
   vs. what we saw on Zigpoll's tiered pricing? What objections does the
   pricing page preemptively address (e.g., "no per-SKU fees," "no credit
   card required")?

5. **Gaps relevant to our differentiation** — confirm whether they show any
   per-forecast explanation/reasoning (not just aggregate accuracy tracking
   after the fact). Note anything that looks like it addresses the "why is
   this forecast what it is" problem our SHAP-based explanation panel is
   built to solve. This is our core differentiator — flag anything that
   undermines that claim so we can address it honestly.

6. **Visual/UX patterns worth adapting** — layout choices, how they visualize
   forecasts (chart types, data density), how technical jargon vs. plain
   language is balanced.

Do NOT copy their copy, layout code, or design assets directly — this is
for understanding patterns and identifying gaps, not replication. Summarize
findings as a short report, then propose 3-5 concrete changes to our
Phase 5 (Frontend Dashboard) and Phase 7 (SEO Landing Page) prompts in
design-doc.md based on what you learn.
