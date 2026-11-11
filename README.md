# CardWise: Credit Card Deep Research

## 1. Project and Target User

CardWise is a multi-agent deep research system that compares U.S. credit cards for a specific person's spending. It combines a 201-card catalog, the CFPB credit card survey and current official issuer pages, then writes a report with a source for every figure.

**Target user:** U.S. consumers choosing between cards who want to see how caps, fees and interest affect what they earn, instead of the advertised rate alone.

## 2. Data Sources

| Source | How it is loaded | What it provides |
|---|---|---|
| [67.CREDIT card dataset](https://67.credit/data) (CC BY 4.0) | Downloaded from the public API `https://67.credit/api/v1/cards.json` and stored as a JSON snapshot in `data/` | 201 cards: issuer, availability, annual fee, category rewards rates, point values, dated welcome offers |
| [CFPB Terms of Credit Card Plans survey](https://www.consumerfinance.gov/data-research/credit-card-data/terms-credit-card-plans-survey/), 2025-12-31 | Downloaded as the published Excel workbook and converted to a CSV of all 663 products in `data/` | Historical purchase and intro APRs, late, foreign-transaction and balance-transfer fees, grace periods |
| Official issuer websites | Searched at question time through Vertex AI Gemini with Google Search grounding | Current rates, caps, fees and terms, kept only when the cited page is on the issuer's own domain |

Download dates and file hashes are recorded in `data/research_sources.json`.

## 3. How It Works

| Agent | Job |
|---|---|
| **Lead** | Asks a clarifying question when the request is too vague, picks the cards to research, assigns research tasks, and decides when the evidence is enough |
| **Rewards Researcher** | Screens the 67.CREDIT catalog for candidates and looks up rewards rates and welcome offers |
| **Costs Researcher** | Looks up historical APRs and fees in the CFPB survey and models borrowing costs |
| **Issuer Verification Researcher** | Finds current terms on official issuer pages and calculates cash back from them |
| **Writer** | Writes the final report from the collected evidence, with inline citations |

- For a multi-card question, the Lead starts only the Researchers relevant to the request. Rewards handles discovery and reward structures, Costs runs for borrowing or fee questions, and Issuer Verification checks current terms and verified reward calculations. Selected Researchers run concurrently in fresh contexts.
- When a material, actionable gap remains, the Lead can send a targeted follow-up, up to 3 formal research rounds and 10 Researcher calls. It stops early when a follow-up adds no new normalized decision field or successful calculation instead of consuming the remaining budget automatically.
- The web page reports leaf-level **Research actions** separately from **Agent runs**. Lifecycle and orchestration events remain visible in the timeline without inflating the research-action count, and a side panel shows every agent's full context.
- `POST /chat` takes `{"message", "session_id"}` and returns `{"response", "session_id", "tool_calls"}`, the same shape as Project 1. `tool_calls` lists every tool call made by the Lead and the Researchers, each with an `agent` field. Reusing a `session_id` continues the conversation, for example to answer a clarifying question.
- The baseline answers below come from the same model as the Writer (`gemini-3.5-flash`) with no tools and no system prompt.

## 4. Quickstart

Prerequisites: Python 3.10 or newer, [`uv`](https://docs.astral.sh/uv/), a Google Cloud project with Vertex AI enabled, and credentials that can call Vertex AI and grounded Google Search.

```bash
uv sync
gcloud auth application-default login
```

Create a `.env` file in the repository root with your project and region:

```dotenv
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
```

The default Lead and Writer model is `vertex_ai/gemini-3.5-flash`; Researchers and the Lead's capacity fallback use `vertex_ai/gemini-3.5-flash-lite`. You can override them with `CARDWISE_LEAD_MODEL`, `CARDWISE_LEAD_FALLBACK_MODEL`, `CARDWISE_RESEARCH_MODEL`, `CARDWISE_WRITER_MODEL`, and `CARDWISE_SEARCH_MODEL`.

Start the web app, then open [http://127.0.0.1:8000](http://127.0.0.1:8000):

```bash
uv run python app.py
```

Run the test suite:

```bash
uv run python -m unittest discover -s tests
```

To reproduce a no-tools baseline for a prompt:

```bash
uv run python compare_baseline.py "Your question here" --output baseline_output/baseline.json
```

## 5. Correctness Gates

CardWise does not treat fluent Researcher or Writer prose as proof. The pipeline applies these gates before presenting a recommendation:

- **Structured evidence is authoritative.** Current product claims must be attached to collected evidence, and a reward total counts as verified only when the structured calculator returns a successful estimate for that card. Researcher summaries cannot override failed searches, missing fields, extraction errors, or incomplete calculations.
- **Winner claims require comparable results.** The Writer may declare an overall winner only when research is complete and every card that requires a calculation has a comparable verified estimate. If at least two candidates are complete, it may give a result explicitly scoped to those verified candidates while naming excluded cards; it cannot turn that into a market-wide “best” claim.
- **Incomplete cards cannot be filled in manually.** A card with an incomplete calculation cannot receive a Writer-invented annual total. Simple arithmetic is allowed only when the decision gate permits it and every input is present in cited evidence.
- **Per-card reporting limits are enforced.** Unknown caps cannot be described as unlimited, unsupported point values cannot be converted to cash, dated catalog facts cannot be presented as current issuer terms, and preliminary catalog screening scores cannot be reported as final net rewards.
- **The final draft is checked again.** A draft that violates the winner or evidence gates is rewritten once with the structured constraints. If the rewrite still fails, CardWise returns a deterministic evidence-limited report instead of the unsafe draft.
- **Research stops on bounded evidence, not just budget exhaustion.** The Lead uses at most 3 formal rounds and 10 Researcher calls, stops early after a no-progress follow-up or a terminal evidence path, and preserves unresolved material gaps rather than retrying the same failure indefinitely.

These gates reduce unsupported certainty; they do not guarantee that issuer pages are available or that an open-market search has found every possible card. The final report must state when missing evidence could change the answer.

## 6. Sample Queries (Needs 2-3 minutes per query to finish the deep research)

### Query 1: Comparing two cards with a reward cap

> I spend $800 per month on groceries, pay in full, and can accept a $95 annual fee. Compare Amex Blue Cash Preferred and Blue Cash Everyday.

**Expected answer:** Blue Cash Preferred should be recommended, by roughly $85 a year once its annual fee is paid. The answer should recognize that both cards cap their higher supermarket rate at $6,000 a year, apply that cap to the user's grocery spending, and show that Preferred's higher grocery rate still outweighs its $95 fee at this level of spending. It should also separate the first year, when the fee is waived, from later years, and distinguish current issuer terms from dated catalog fee data.

**Where it comes from:**

- Supermarket rates, the $6,000 cap and the first-year fee waiver: the American Express product pages for [Blue Cash Preferred](https://www.americanexpress.com/us/credit-cards/card/blue-cash-preferred/) and [Blue Cash Everyday](https://www.americanexpress.com/us/credit-cards/card/blue-cash-everyday/).
- Annual fees: 67.CREDIT dataset snapshots dated May 2026, records `blue-cash-preferred` and `blue-cash-everyday`, field `annualFee`; the Preferred introductory first-year fee is also supported by its issuer page.
- Reward Dollar redemption and current product restrictions: the American Express product pages.

**Agent vs. baseline:**

<details>
<summary><h3>CardWise agent answer (click to expand)</h3></summary>

For a monthly grocery spend of $800 ($9,600 annually), the **American Express Blue Cash Preferred® Card** is the better option.

Even after accounting for its $95 ongoing annual fee, the Blue Cash Preferred card delivers an estimated ongoing net reward of **$301 per year** on your groceries alone, compared to **$216 per year** from the no-annual-fee Blue Cash Everyday® Card. That is an ongoing advantage of **$85 per year**. In your first year of card ownership, your net earnings with the Blue Cash Preferred card rise to **$396** because the annual fee is waived for the first year.

### Comparison Overview

Here is how the grocery rewards and fees compare for both cards based on your grocery spending of $9,600 per year:

| Feature | Amex Blue Cash Everyday | Amex Blue Cash Preferred |
| :--- | :--- | :--- |
| **Grocery Reward Rate** | 3% cash back on eligible U.S. supermarkets ([source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-everyday/)) | 6% cash back on eligible U.S. supermarkets ([source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-preferred/)) |
| **Annual Limit for Premium Rate** | First $6,000 per calendar year ([source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-everyday/)) | First $6,000 per calendar year ([source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-preferred/)) |
| **Over-Cap Reward Rate** | 1% cash back ([source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-everyday/)) | 1% cash back ([source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-preferred/)) |
| **First-Year Annual Fee** | $0 ([source](https://67.credit/card/blue-cash-everyday), dated May 2026) | $0 introductory annual fee ([source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-preferred/)) |
| **Ongoing Annual Fee** | $0 ([source](https://67.credit/card/blue-cash-everyday), dated May 2026) | $95 ([source](https://67.credit/card/blue-cash-preferred), dated May 2026) |
| **Annual Grocery Cash Back (Gross)** | $216 | $396 |
| **First-Year Net Grocery Value** | **$216** | **$396** |
| **Ongoing Net Grocery Value** | **$216** | **$301** |

### Detailed Reward Calculations

Because both cards cap their premium grocery rewards at $6,000 per calendar year, your annual spending of $9,600 is split into two tiers for both cards: $6,000 at the premium rate and the remaining $3,600 at the 1% over-cap rate.

#### Amex Blue Cash Everyday

* **Premium-rate earnings:** $6,000 × 3% = $180
* **Over-cap earnings:** $3,600 × 1% = $36
* **Total gross cash back:** $180 + $36 = $216
* **Ongoing net cash back:** **$216** after the $0 annual fee

#### Amex Blue Cash Preferred

* **Premium-rate earnings:** $6,000 × 6% = $360
* **Over-cap earnings:** $3,600 × 1% = $36
* **Total gross cash back:** $360 + $36 = $396
* **First-year net cash back:** **$396** after the $0 introductory annual fee
* **Ongoing net cash back:** $396 − $95 = **$301**

### Redemptions and Restrictions

* **Supermarket definition:** The elevated rates apply to eligible purchases at U.S. supermarkets ([source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-preferred/)). Superstores and warehouse clubs may not qualify for the elevated supermarket rate, so where you buy groceries can materially change the result.
* **Redemption method:** Cash back is earned as Reward Dollars and can be redeemed using the methods described in the current American Express terms ([source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-preferred/)).

</details>

<details>
<summary><h3>Baseline answer, no tools and no system prompt (click to expand)</h3></summary>

Based on your monthly grocery spend of $800 ($9,600 annually) and your willingness to pay a $95 annual fee, **the Blue Cash Preferred (BCP) is the clear winner**, even with its $6,000 annual cap on the highest cash-back tier. 

Because you pay in full every month, you do not need to worry about interest rates, allowing you to focus purely on maximizing your cash-back yield.

Here is the exact mathematical breakdown and a comparison of the other benefits of both cards to help you decide.

---

### The Math: Grocery Spend Only ($9,600/year)

Both cards cap their elevated grocery rewards at **$6,000 per calendar year**, after which the reward rate drops to 1%. Because your annual spend is $9,600, you will hit this cap on both cards.

#### 1. Amex Blue Cash Preferred (BCP)
*   **Annual Fee:** $95 (often waived the first year, but we will assume $95 for long-term calculations).
*   **First $6,000 at 6%:** $360
*   **Remaining $3,600 at 1%:** $36
*   **Total Grocery Cash Back:** $396
*   **Net Cash Back (after $95 fee):** **$301**

#### 2. Amex Blue Cash Everyday (BCE)
*   **Annual Fee:** $0
*   **First $6,000 at 3%:** $180
*   **Remaining $3,600 at 1%:** $36
*   **Total Grocery Cash Back:** $216
*   **Net Cash Back:** **$216**

#### **Grocery Winner:** Blue Cash Preferred (by $85 per year)

---

### Beyond Groceries: Card Comparison

To make the final decision, you should look at how the rest of your monthly spending aligns with the other categories on these cards.

| Feature | Blue Cash Preferred (BCP) | Blue Cash Everyday (BCE) |
| :--- | :--- | :--- |
| **Annual Fee** | $95 (usually $0 first year) | $0 |
| **U.S. Supermarkets** | **6%** (up to $6k/yr, then 1%) | **3%** (up to $6k/yr, then 1%) |
| **U.S. Online Retail** | 1% | **3%** (up to $6k/yr, then 1%) |
| **Gas / Fuel** | **3%** (uncapped) | **3%** (up to $6k/yr, then 1%) |
| **Transit (Tolls, Rideshare, Parking, Trains)** | **3%** (uncapped) | 1% |
| **Streaming Subscriptions** | **6%** (uncapped) | 1% |

#### Why you might still choose the Blue Cash Preferred (BCP):
*   **Streaming & Transit:** If you pay for streaming services (Netflix, Spotify, Disney+, etc.) or commute via tolls, trains, or rideshares, the BCP’s uncapped 6% on streaming and 3% on transit will widen the gap even further in favor of the BCP.
*   **Uncapped Gas:** If you spend heavily on gas, the BCP has no cap on its 3% gas tier, whereas the BCE caps it at $6,000 per year.

#### Why you might choose the Blue Cash Everyday (BCE) instead:
*   **Online Shopping:** The BCE offers 3% back on U.S. online retail (Amazon, clothing websites, target.com, etc., up to $6,000/year). The BCP only offers 1% on these purchases. If you do a massive amount of online shopping, the BCE could theoretically close the $85 gap.

---

### Additional Credits (Both Cards)

Both cards offer statement credits that can easily offset the BCP's annual fee entirely:
*   **Disney Bundle Credit:** Both cards offer a **$7/month credit** ($84/year) when you spend $9.99 or more each month on a Disney Bundle subscription. 
*   **BCP Only:** Offers up to $120/year in Equinox credits ($10/month).
*   **BCE Only:** Offers up to $180/year in Home Chef credits ($15/month).

### Summary Recommendation

Buy the **Amex Blue Cash Preferred**. 

Even with the $95 annual fee and the $6,000 grocery cap, your $800/month grocery spend makes the BCP **$85 more profitable** than the BCE. If you also use the BCP for streaming (6%), transit (3%), or use the $84 Disney Bundle credit, the card pays for its annual fee several times over.

</details>

**Comment:** Both answers reach the same recommendation and ongoing figures: $301 for Blue Cash Preferred versus $216 for Blue Cash Everyday, a difference of $85 per year. The CardWise answer also separates the introductory first year from later years, shows the capped and over-cap calculations, and links the decision inputs to issuer or dated catalog sources. The baseline provides no sources and adds statement-credit details that may be out of date.

### Query 2: Balance transfer costs

> I want to transfer a $5,000 balance and pay it off over 18 months. Compare the BankAmericard and the PenFed Gold Visa: the transfer fee, how long the 0% intro APR lasts, and the APR afterward.

**Expected answer:** The PenFed Gold Visa should come out as the lower-cost option for either a faster 15-month payoff or the requested 18-month plan. The answer should use each issuer's current balance transfer terms: BankAmericard's 21-billing-cycle 0% period covers the full plan but charges a 5% transfer fee, while PenFed charges 3% and offers 0% for 15 months. It should show that PenFed saves $100 if paid off within 15 months. For an 18-month payoff with the transfer fee added to the balance, an equal-payment estimate is about $287.51 per month, with about $25 of post-intro interest and about $175 in total cost—roughly $75 less than BankAmericard. Actual statement timing and daily interest can shift the final payment slightly.

**Where it comes from:**

- Current transfer fees, intro periods, APRs and annual fees: the issuer pages for the [BankAmericard](https://www.bankofamerica.com/credit-cards/products/bankamericard-credit-card/) and the [PenFed Gold Visa](https://www.penfed.org/credit-cards/gold-visa).

**Agent vs. baseline:**

<details>
<summary><h3>CardWise agent answer (click to expand)</h3></summary>

For a $5,000 balance transfer paid off over 18 months, the **PenFed Gold Visa is expected to have the lower total cost**, while the **BankAmericard is the safer option if you want the entire repayment period covered by a 0% introductory APR**.

BankAmericard's 21-billing-cycle introductory period covers the full 18-month plan but charges a 5% transfer fee. PenFed charges a lower 3% fee, but its 0% period lasts only 15 months, so the remaining balance accrues interest for approximately three months.

Neither card charges an annual fee ([BankAmericard source](https://www.bankofamerica.com/credit-cards/products/bankamericard-credit-card/), [PenFed source](https://www.penfed.org/credit-cards/gold-visa)).

---

### Comparison Summary

| Feature | BankAmericard | PenFed Gold Visa |
| :--- | :--- | :--- |
| **Introductory APR Duration** | **0% APR for 21 billing cycles** on transfers made in the first 60 days ([source](https://www.bankofamerica.com/credit-cards/products/bankamericard-credit-card/)) | **0% APR for 15 months** on transfers made in the first 60 days ([source](https://www.penfed.org/credit-cards/gold-visa)) |
| **Balance Transfer Fee** | **5%** of the transfer amount ([source](https://www.bankofamerica.com/credit-cards/products/bankamericard-credit-card/)) | **3%** of the transfer amount ([source](https://www.penfed.org/credit-cards/gold-visa)) |
| **Upfront Fee on $5,000** | **$250** | **$150** |
| **Post-Introductory APR** | **15.24% to 26.24% variable** ([source](https://www.bankofamerica.com/credit-cards/products/bankamericard-credit-card/)) | **17.99% variable**, based on the Prime Rate ([source](https://www.penfed.org/credit-cards/gold-visa)) |
| **Annual Fee** | $0 ([source](https://www.bankofamerica.com/credit-cards/products/bankamericard-credit-card/)) | $0 ([source](https://www.penfed.org/credit-cards/gold-visa)) |

---

### How the Numbers Work Out for an 18-Month Payoff

#### Scenario A: BankAmericard (Full 18-Month Interest-Free Coverage)
If you transfer $5,000 to the BankAmericard:
1. **Upfront Fee:** $5,000 × 5% = $250 balance transfer fee ([source](https://www.bankofamerica.com/credit-cards/products/bankamericard-credit-card/)).
2. **Total Starting Balance:** $5,250, assuming the transfer fee is added to the card balance.
3. **Interest Incurred:** $0. Because the 0% introductory period lasts for 21 billing cycles ([source](https://www.bankofamerica.com/credit-cards/products/bankamericard-credit-card/)), your entire 18-month payoff period is covered.
4. **Monthly Payment:** To pay it off exactly in 18 months, you would pay **about $291.67 per month** ($5,250 / 18).
5. **Total Cost:** **$250**.

#### Scenario B: PenFed Gold Visa (Lower Expected Total Cost)
If you transfer $5,000 to the PenFed Gold Visa:
1. **Upfront Fee:** $5,000 × 3% = $150 balance transfer fee ([source](https://www.penfed.org/credit-cards/gold-visa)).
2. **Total Starting Balance:** $5,150, assuming the transfer fee is added to the card balance.
3. **The 15-Month Limit:** The 0% introductory period only lasts for 15 months ([source](https://www.penfed.org/credit-cards/gold-visa)).
4. **Your Payoff Strategy Options:**
   * **Option 1: Pay it off in 15 months (Recommended).** If you increase your monthly budget to pay **about $343.33 per month** ($5,150 / 15), you will pay off the balance before the intro period ends. In this case, you incur $0 interest. **Total Cost: $150** (saving you $100 compared to BankAmericard).
   * **Option 2: Pay it off over 18 months.** Simply dividing the $5,150 starting balance by 18 gives **$286.11**, but that is only a zero-interest base payment and would leave a small residual balance once post-intro interest is added. Using a monthly rate of 17.99% / 12 after month 15, an equal-payment estimate is **about $287.51 per month for 18 months**. That produces approximately **$25.23 of interest** and a total transfer cost of about **$175.23** ($150 fee plus estimated interest). Alternatively, you could pay $286.11 initially and increase the last three payments enough to cover the accrued interest. The exact final payment depends on statement timing, daily interest, and payment allocation, but either approach keeps the estimated cost below BankAmericard's $250 fee.

---

### Key Card Restrictions to Consider
* **Exclusions:** BankAmericard does not allow you to use balance transfers to pay off other accounts held with Bank of America ([source](https://www.bankofamerica.com/credit-cards/products/bankamericard-credit-card/)).
* **New purchases on PenFed:** PenFed states that new purchases can begin accruing interest while you carry the transferred balance unless you pay the entire balance, including the transfer, by the due date ([source](https://www.penfed.org/credit-cards/gold-visa)). Avoid using the PenFed card for new spending during the payoff period.

### Sources
* [BankAmericard Terms and Conditions](https://www.bankofamerica.com/credit-cards/products/bankamericard-credit-card/) (As of November 11, 2026)
* [PenFed Gold Visa Terms and Conditions](https://www.penfed.org/credit-cards/gold-visa) (As of November 11, 2026)

</details>

<details>
<summary><h3>Baseline answer, no tools and no system prompt (click to expand)</h3></summary>

To pay off a $5,000 balance over 18 months, comparing the **BankAmericard® credit card** and the **PenFed Gold Visa® Card** is a smart move. Because you have a strict 18-month timeline, the length of the 0% introductory APR period is the most critical factor. 

Here is a side-by-side comparison of how these two cards stack up for your specific scenario.

### Side-by-Side Comparison

| Feature | BankAmericard® Credit Card | PenFed Gold Visa® Card |
| :--- | :--- | :--- |
| **Intro Balance Transfer APR** | **0%** | **0%** |
| **Intro Period Length** | **18 billing cycles** (for transfers made within the first 60 days) | **15 months** (for transfers made within the promotional window) |
| **Balance Transfer Fee** | **3%** of the transfer amount (minimum $10) | **3%** of the transfer amount |
| **Post-Intro Ongoing APR** | **16.24% - 26.24% (Variable)** (based on creditworthiness) | **14.49% - 17.99% (Variable)** (based on creditworthiness) |
| **Annual Fee** | $0 | $0 |
| **Membership Required?** | No | Yes (must join PenFed Credit Union; free to join with a $5 savings account deposit) |

---

### How the $5,000 Transfer Plays Out (18-Month Plan)

Here is exactly how much you would pay with each card if you stick to your goal of paying off the $5,000 over 18 months.

#### Option 1: BankAmericard® (The Better Fit for 18 Months)
Because the 0% intro APR matches your 18-month goal, you can pay off the balance entirely interest-free.
*   **Balance Transfer Fee:** $150 (3% of $5,000)
*   **Total Debt to Pay:** $5,150
*   **Monthly Payment for 18 Months:** **$286.11**
*   **Total Interest Paid:** **$0**

#### Option 2: PenFed Gold Visa® 
Because the 0% intro APR only lasts for 15 months, you have two choices under this card:

*   **Scenario A: Pay it off in 15 months to avoid interest.**
    *   **Balance Transfer Fee:** $150 (3% of $5,000)
    *   **Total Debt to Pay:** $5,150
    *   **Monthly Payment for 15 Months:** **$343.33**
    *   **Total Interest Paid:** **$0**

*   **Scenario B: Stick strictly to your 18-month timeline.**
    If you pay the same $286.11 per month, you will still have a remaining balance of **$858.35** when the 15-month intro period ends. 
    *   For months 16, 17, and 18, that remaining balance will accrue interest at the regular ongoing APR (14.49% to 17.99% variable). 
    *   **Total Interest Paid:** You would pay roughly **$20 to $35 in interest** over those final three months (depending on your assigned APR).

---

### Key Takeaways & Recommendation

*   **For your 18-month timeline, the BankAmericard is the winner.** It allows you to spread your payments evenly over the full 18 months ($286.11/month) without paying a single penny in interest beyond the upfront $150 transfer fee. 
*   **The PenFed Gold Visa is better for long-term borrowing.** While its 15-month 0% period is slightly too short for your timeline, its ongoing regular APR (14.49% - 17.99%) is much lower than Bank of America’s top-tier rate (up to 26.24%). If you think you might carry a balance *after* the 18 months, PenFed is the safer long-term card. 

*Note: To qualify for the introductory rates on either card, you must initiate the balance transfer within the timeframe specified in your cardholder agreement (usually 60 days from account opening).*

</details>

**Comment:** Here the answers disagree because the baseline uses outdated BankAmericard terms: a 3% transfer fee and an 18-cycle introductory period. The CardWise answer uses the current issuer terms—a 5% BankAmericard fee and 21 billing cycles—and shows the trade-off explicitly. PenFed costs $150 if paid off in 15 months and about $175 on an equal-payment 18-month estimate, versus BankAmericard's $250 fee, so PenFed remains the lower-cost option despite three months of interest.

### Query 3: Is a recommended card the right long-term choice?

> I spend about $700 per month on dining and takeout, $300 on entertainment, and $200 at grocery stores. I always pay my balance in full and want to avoid an annual fee. A friend recommended the Capital One Savor card. Is it the right long-term choice compared with the Wells Fargo Active Cash card, and how much monthly spending outside Savor’s bonus categories would make Active Cash the better option? Ignore welcome bonuses.

**Expected answer:** Capital One Savor should be the better long-term choice for the stated spending, earning about $432 per year versus $288 from Wells Fargo Active Cash. The answer should calculate each category separately, ignore welcome bonuses, and show that Savor's $12 monthly lead is offset when spending outside its 3% categories exceeds $1,200 per month because Active Cash earns 2% rather than Savor's 1% on that spending.

**Where it comes from:**

- Savor's dining, entertainment, grocery and base rates, merchant exclusions, and annual fee: the [Capital One Savor product page](https://www.capitalone.com/credit-cards/savor/) and [Capital One's Savor category guide](https://www.capitalone.com/learn-grow/money-management/everything-about-savor/).
- Active Cash's flat 2% rate: the [Wells Fargo Active Cash product page](https://creditcards.wellsfargo.com/active-cash-credit-card/).
- Active Cash's $0 annual fee: 67.CREDIT catalog snapshot dated December 6, 2025, record `wf-active-cash`, field `annualFee`.

**Agent vs. baseline:**

<details>
<summary><h3>CardWise agent answer (click to expand)</h3></summary>

For your spending of $700 per month on dining, $300 on entertainment, and $200 at grocery stores, the **Capital One Savor is the better long-term choice of these two cards**.

Savor earns an estimated **$432 per year**, compared with **$288 per year** from Wells Fargo Active Cash. Active Cash becomes the better option only when spending outside Savor's 3% categories exceeds **$1,200 per month**, assuming your current dining, entertainment, and grocery spending remains unchanged and qualifies for the stated reward categories.

### Reward Comparison on Your Monthly Spending

Both cards carry a $0 annual fee. Capital One lists Savor's ongoing annual fee as $0 ([source](https://www.capitalone.com/credit-cards/savor/)), while the Active Cash $0 fee comes from a 67.CREDIT catalog snapshot dated December 6, 2025 ([source](https://67.credit/card/wf-active-cash)).

| Spending Category | Monthly Spending | Capital One Savor | Wells Fargo Active Cash® |
| :--- | :--- | :--- | :--- |
| **Dining and Takeout** | $700 | 3% cash back ([source](https://www.capitalone.com/learn-grow/money-management/everything-about-savor/)); **$252/year** | 2% cash rewards ([source](https://creditcards.wellsfargo.com/active-cash-credit-card/)); **$168/year** |
| **Entertainment** | $300 | 3% cash back ([source](https://www.capitalone.com/learn-grow/money-management/everything-about-savor/)); **$108/year** | 2% cash rewards ([source](https://creditcards.wellsfargo.com/active-cash-credit-card/)); **$72/year** |
| **Grocery Stores** | $200 | 3% cash back ([source](https://www.capitalone.com/learn-grow/money-management/everything-about-savor/)); **$72/year** | 2% cash rewards ([source](https://creditcards.wellsfargo.com/active-cash-credit-card/)); **$48/year** |
| **Total Modeled Spending** | **$1,200/month** | **$432/year** | **$288/year** |

Capital One excludes superstores such as Walmart and Target from Savor's grocery category ([source](https://www.capitalone.com/learn-grow/money-management/everything-about-savor/)). Purchases at those merchants may earn Savor's base rate rather than 3%, which would reduce its advantage over Active Cash.

### Break-Even Point for Other Spending

Savor earns 1% on purchases outside its bonus categories ([source](https://www.capitalone.com/credit-cards/savor/)), while Active Cash earns 2% on eligible net purchases ([source](https://creditcards.wellsfargo.com/active-cash-credit-card/)). Active Cash therefore gains one cent for every dollar of other spending.

Savor currently leads by `$432 − $288 = $144 per year`, or `$12 per month`. The break-even calculation is:

`$12 ÷ 1% = $1,200 of other spending per month`

* **Less than $1,200 per month in other spending:** Savor produces more total cash back.
* **Exactly $1,200 per month:** Both cards earn $48 per month in total rewards.
* **More than $1,200 per month:** Active Cash produces more total cash back.

For example, with $500 per month of other spending, Savor earns $41 per month and Active Cash earns $34. At $1,500 of other spending, Active Cash earns $54 per month and Savor earns $51.

As long as your dining, entertainment, and eligible grocery spending remains similar and your other spending stays below $1,200 per month, Savor is the stronger long-term fit of these two cards.

</details>

<details>
<summary><h3>Baseline answer, no tools and no system prompt (click to expand)</h3></summary>

To determine whether the **Capital One Savor** (formerly SavorOne) or the **Wells Fargo Active Cash** is the better long-term choice for you, we can break this down with direct math based on your specific monthly spending.

### 1. Comparing Earnings on Your Current Spend
Your specified monthly budget is **$1,200**:
*   **Dining/Takeout:** $700
*   **Entertainment:** $300
*   **Grocery Stores:** $200

Here is how much cashback you would earn on this spend with each card:

| Card | Category Multipliers | Monthly Earnings | Yearly Earnings |
| :--- | :--- | :--- | :--- |
| **Capital One Savor** | 3% on Dining, Entertainment, & Groceries | ($1,200 × 3%) = **$36.00** | **$432.00** |
| **Wells Fargo Active Cash** | 2% flat-rate on all purchases | ($1,200 × 2%) = **$24.00** | **$288.00** |

On your core spending alone, the **Capital One Savor wins by $12 per month ($144 per year)**.

---

### 2. The Breakeven Point: How much "Other" spending favors Active Cash?
The Wells Fargo Active Cash earns 2% on non-category ("other") spending, while the Savor only earns 1%.

To find the exact amount of monthly "other" spending ($X) where the Active Cash catches up and becomes the better option, we set up a simple algebraic equation:

$$\text{Savor Earnings} = \text{Active Cash Earnings}$$
$$\$36 + 0.01X = \$24 + 0.02X$$
$$\$12 = 0.01X$$
$$X = \$1,200$$

*   **If you spend less than $1,200/month** on non-bonus categories (e.g., retail, gas, utilities, insurance, auto repairs), the **Capital One Savor** is the better card.
*   **If you spend exactly $1,200/month** on non-bonus categories, the cards yield the exact same cashback ($48/month).
*   **If you spend more than $1,200/month** on non-bonus categories, the **Wells Fargo Active Cash** is the better card overall.

---

### 3. Key Nuances to Consider for the "Long-Term"
While the math is straightforward, there are a few real-world rules that could tip the scale:

*   **Where do you buy groceries?** Capital One defines "groceries" strictly. Supermarkets like Kroger, Aldi, and Whole Foods earn 3%. However, superstores (Walmart, Target) and wholesale clubs (Costco, Sam's Club) **do not** qualify for 3% and will only earn 1%. If you do your grocery shopping at Walmart or Costco, that $200 grocery spend drops to 1% on the Savor, lowering your Savor monthly earnings to $32 and reducing your non-bonus breakeven threshold to **$800/month**.
*   **Do you travel internationally?** The Capital One Savor has **no foreign transaction fees**. The Wells Fargo Active Cash charges a **3% fee** on foreign purchases. If you travel outside the U.S. or buy from international online merchants, the Savor is the clear winner.
*   **Future Upgrade Path:** If you ever decide to get into travel rewards later, the Savor is highly valuable. You can transfer your Savor cashback into Capital One miles if you eventually pair it with a premium card like the Capital One Venture X. Wells Fargo has a similar ecosystem with the Autograph Journey, but Capital One's transfer partners are generally considered stronger.

### Summary Verdict
If your monthly non-bonus spending (gas, bills, shopping, etc.) is **under $1,200**, the **Capital One Savor** is the superior, more lucrative long-term choice.

*Tip: Because both cards have $0 annual fees, the absolute best long-term strategy is to get **both**. Use the Savor to get 3% on dining, entertainment, and groceries, and use the Active Cash as a "catch-all" to get 2% on everything else.*

</details>

**Comment:** Both answers reach the same core result: Savor earns $432 per year on the stated categories, Active Cash earns $288, and $1,200 per month of other spending is the break-even point. CardWise grounds the rates, annual fees, merchant exclusions and calculations in issuer or dated catalog evidence and limits the recommendation to the two requested cards. The baseline has no sources and adds foreign-transaction-fee, transfer-partner and two-card-strategy claims that were not requested or verified during the comparison.

### Vague questions

> What's the best credit card for me?

CardWise does not start research when a question lacks the details that decide the answer. For this question, the Lead replies:

> To help me find the best credit card for you, could you please share:
> 1. Your typical **monthly spending** by category (e.g., groceries, dining, gas, travel, utilities)?
> 2. Your preference regarding **annual fees** (e.g., strictly no-fee, or open to annual fees for better rewards)?
> 3. Whether you are interested in **business cards**, or only personal cards?

No Researcher or Writer runs for this turn. When the user answers, the earlier question and the Lead's request are passed back with the reply, and research begins.

