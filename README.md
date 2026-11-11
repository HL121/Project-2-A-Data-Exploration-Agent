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

- For a multi-card question, the Lead starts all three Researchers at once, each in a fresh context.
- When a finding leaves a gap, the Lead sends a targeted follow-up, up to 3 research rounds and 10 Researcher calls.
- The web page shows each agent's steps as they happen, and a side panel shows every agent's full context.
- `POST /chat` takes `{"message", "session_id"}` and returns `{"response", "session_id", "tool_calls"}`, the same shape as Project 1. `tool_calls` lists every tool call made by the Lead and the Researchers, each with an `agent` field. Reusing a `session_id` continues the conversation, for example to answer a clarifying question.
- The baseline answers below come from the same model as the Writer (`gemini-3.5-flash`) with no tools and no system prompt.

## 4. Sample Queries (Needs 2-3 minutes per query to finish the deep research)

### Query 1: Comparing two cards with a reward cap

> I spend $800 per month on groceries, pay in full, and can accept a $95 annual fee. Compare Amex Blue Cash Preferred and Blue Cash Everyday.

**Expected answer:** Blue Cash Preferred should be recommended, by roughly $85 a year once its annual fee is paid. The answer should recognize that both cards cap their higher supermarket rate at $6,000 a year, apply that cap to the user's grocery spending, and show that Preferred's higher grocery rate still outweighs its $95 fee at this level of spending. It should also separate the first year, when the fee is waived, from later years, and cite the issuer's current terms.

**Where it comes from:**

- Supermarket rates, the $6,000 cap and the first-year fee waiver: the American Express product pages for [Blue Cash Preferred](https://www.americanexpress.com/us/credit-cards/card/blue-cash-preferred/) and [Blue Cash Everyday](https://www.americanexpress.com/us/credit-cards/card/blue-cash-everyday/).
- Annual fees and welcome offers: 67.CREDIT dataset, records `blue-cash-preferred` and `blue-cash-everyday`, fields `annualFee` and `currentBonus`.
- Purchase APRs and grace period: CFPB survey, the American Express Blue Cash Preferred and Blue Cash Everyday rows.

**Agent vs. baseline:**

<table>
<tr><th width="50%">CardWise agent</th><th width="50%">Baseline (no tools, no system prompt)</th></tr>
<tr><td valign="top">

For your spending habits, the **American Express Blue Cash Preferred® Card** is the better option. Even after accounting for its $95 annual fee, the card yields **$301 in net cash back** annually on groceries alone, which is $85 more than you would earn with the no-fee Blue Cash Everyday® Card.

### Grocery Cash Back Comparison

With $800 per month in grocery spending, you spend $9,600 annually. Both cards cap their highest rewards rate at $6,000 per calendar year, reverting to 1% cash back on any supermarket spending beyond that cap. 

| Feature | Blue Cash Everyday® Card | Blue Cash Preferred® Card |
| :--- | :--- | :--- |
| **Annual Fee** | $0 ([Source](https://67.credit/card/blue-cash-everyday), 2026-05-12) | $95 (Ongoing) ([Source](https://67.credit/card/blue-cash-preferred), 2026-05-12)<br>$0 (Introductory Year 1) ([Source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-preferred/), 2026-11-11) |
| **Grocery Reward Rate** | 3% cash back ([Source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-everyday/), 2026-11-11) | 6% cash back ([Source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-preferred/), 2026-11-11) |
| **Annual Cap on Grocery Rate** | First $6,000 in purchases ([Source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-everyday/), 2026-11-11) | First $6,000 in purchases ([Source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-preferred/), 2026-11-11) |
| **Over-Cap Reward Rate** | 1% cash back ([Source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-everyday/), 2026-11-11) | 1% cash back ([Source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-preferred/), 2026-11-11) |
| **Gross Grocery Cash Back** | $216.00 ([Source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-everyday/), 2026-11-11) | $396.00 ([Source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-preferred/), 2026-11-11) |
| **Net Grocery Cash Back (Year 1)** | **$216.00** | **$396.00** ([Source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-preferred/), 2026-11-11) |
| **Net Grocery Cash Back (Ongoing)**| **$216.00** | **$301.00** ([Source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-preferred/), 2026-11-11) |

---

### How the Calculations Work

Here is the step-by-step math for your annual grocery spending of $9,600:

#### 1. Amex Blue Cash Everyday
* **Under-cap rewards:** $6,000 × 3% = $180
* **Over-cap rewards:** $3,600 × 1% = $36
* **Total Gross Cash Back:** $180 + $36 = $216
* **Net Cash Back:** $216 - $0 (Annual Fee) = **$216**

#### 2. Amex Blue Cash Preferred (Ongoing Years)
* **Under-cap rewards:** $6,000 × 6% = $360
* **Over-cap rewards:** $3,600 × 1% = $36
* **Total Gross Cash Back:** $360 + $36 = $396
* **Net Cash Back:** $396 - $95 (Annual Fee) = **$301**
*(Note: During your first year, the annual fee is introductory $0, resulting in a Year 1 net benefit of **$396**).*

---

### Key Differences and Conditions

* **Welcome Offers:** As a first-year extra, the Blue Cash Preferred offers a **$350 welcome offer** ([Source](https://67.credit/card/blue-cash-preferred), 2026-07-29) compared to the Blue Cash Everyday's **$200 offer** ([Source](https://67.credit/card/blue-cash-everyday), 2026-07-29). Note that the specific spending thresholds required to earn these rewards should be confirmed with the issuer.
* **Supermarket Restrictions:** Amex rewards apply strictly to "U.S. supermarkets." Specialty stores, superstores (like Target and Walmart), and warehouse clubs (like Costco or Sam's Club) are generally excluded from the 3% and 6% categories and will only earn the 1% base rate.
* **Other Earning Categories:** If you occasionally spend outside of groceries, the **Blue Cash Everyday** also earns 3% cash back (up to $6,000/year, then 1%) on U.S. online retail and U.S. gas stations ([Source](https://www.americanexpress.com/us/credit-cards/card/blue-cash-everyday/), 2026-11-11). The **Blue Cash Preferred** earns 6% on select streaming services and 3% on U.S. gas stations and transit ([Source](https://67.credit/card/blue-cash-preferred), 2026-05-12).
* **Payment Habits:** Since you pay your bill in full every month, you can easily avoid interest charges by taking advantage of the **25-day purchase grace period** offered on both cards ([Source](https://files.consumerfinance.gov/f/documents/cfpb_tccp-data_2025-12-31.xlsx), 2025-12-31). For historical context, standard purchase APRs for both cards range from a minimum of 18.24% to a maximum of 29.24% depending on your credit profile ([Source](https://files.consumerfinance.gov/f/documents/cfpb_tccp-data_2025-12-31.xlsx), 2025-12-31).

</td><td valign="top">

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

</td></tr>
</table>

**Comment:** Both answers reach the same conclusion and figures, since these are two of the best-known cash back cards. The difference is that every rate, cap and fee in the agent's answer links to a dated source the user can check, while the baseline gives no sources, including for details such as statement credits that may be out of date.

### Query 2: Balance transfer costs

> I want to transfer a $5,000 balance and pay it off over 18 months. Compare the BankAmericard and the PenFed Gold Visa: the transfer fee, how long the 0% intro APR lasts, and the APR afterward.

**Expected answer:** The PenFed Gold Visa should come out as the lower-cost option, saving somewhere around $70 to $80 overall. The answer should use each issuer's current balance transfer terms, under which the BankAmericard's longer 0% period covers the whole plan but its fee is 5% against PenFed's 3%. It should account for the interest PenFed charges after its 15-month intro period ends, point out where the current terms differ from older survey data, and cite its sources.

**Where it comes from:**

- Current transfer fees, intro periods, APRs and annual fees: the issuer pages for the [BankAmericard](https://www.bankofamerica.com/credit-cards/products/bankamericard-credit-card/) and the [PenFed Gold Visa](https://www.penfed.org/credit-cards/gold-visa).
- Earlier BankAmericard terms: CFPB survey (2025-12-31), the BankAmericard row, columns for balance transfer fee, intro balance transfer APR length, and purchase APR.

**Agent vs. baseline:**

<table>
<tr><th width="50%">CardWise agent</th><th width="50%">Baseline (no tools, no system prompt)</th></tr>
<tr><td valign="top">

To pay off a $5,000 balance over 18 months, you can compare the current official terms (as of November 2026) for the **BankAmericard® Credit Card** and the **PenFed Gold Visa® Card**. 

While the BankAmericard offers a longer promotional window that fully covers your 18-month timeline, the PenFed Gold Visa features a lower upfront fee that may make it cheaper overall, even with some interest charges at the end.

### Summary Comparison

| Feature | BankAmericard® Credit Card | PenFed Gold Visa® Card |
| :--- | :--- | :--- |
| **Intro Balance Transfer APR** | 0% | 0% |
| **0% Intro Duration** | 21 billing cycles (approx. 21 months) ([source](https://www.bankofamerica.com/credit-cards/products/bankamericard-credit-card/)) | 15 months ([source](https://www.penfed.org/credit-cards/gold-visa)) |
| **Balance Transfer Fee** | 5% ([source](https://www.bankofamerica.com/credit-cards/products/bankamericard-credit-card/)) | 3% ([source](https://www.penfed.org/credit-cards/gold-visa)) |
| **Upfront Fee on $5,000** | $250 | $150 |
| **Ongoing APR (After Intro)** | 15.24% to 26.24% variable ([source](https://www.bankofamerica.com/credit-cards/products/bankamericard-credit-card/)) | 17.99% variable ([source](https://www.penfed.org/credit-cards/gold-visa)) |
| **Annual Fee** | $0 ([source](https://www.bankofamerica.com/credit-cards/products/bankamericard-credit-card/)) | $0 ([source](https://www.penfed.org/credit-cards/gold-visa)) |

*Note: Historical CFPB survey data from December 31, 2025, previously recorded a 4% balance transfer fee, an 18-month introductory window, and an ongoing APR of 14.49% to 26.74% for the BankAmericard ([source](https://files.consumerfinance.gov/f/documents/cfpb_tccp-data_2025-12-31.xlsx)). The terms listed in the table above represent the current updated rates.*

---

### How the Numbers Compare for Your 18-Month Scenario

#### Scenario A: BankAmericard® Credit Card
* **Upfront Fee:** $5,000 x 5% = $250 ([source](https://www.bankofamerica.com/credit-cards/products/bankamericard-credit-card/)).
* **Interest Charges:** $0. Because the 0% intro period lasts for 21 billing cycles (about 21 months) ([source](https://www.bankofamerica.com/credit-cards/products/bankamericard-credit-card/)), your entire 18-month repayment period is completely interest-free.
* **Total Repayment Cost:** **$5,250** ($5,000 principal + $250 transfer fee). This requires an equal monthly payment of about $291.67 over 18 months.

#### Scenario B: PenFed Gold Visa® Card
* **Upfront Fee:** $5,000 x 3% = $150 ([source](https://www.penfed.org/credit-cards/gold-visa)).
* **Interest Charges:** The 0% intro period only lasts 15 months ([source](https://www.penfed.org/credit-cards/gold-visa)). This means the remaining 3 months of your 18-month plan will accrue interest at the ongoing variable APR of 17.99% ([source](https://www.penfed.org/credit-cards/gold-visa)). 
* **Total Repayment Cost Estimate:** If you make equal monthly payments of about $286.11 to pay off the total balance ($5,150) over 18 months, you will have a remaining balance of about $858.35 after month 15. Over the final 3 months, this remaining balance will accrue roughly $26 in interest. This brings your total estimated payoff cost to **about $5,176**. 

### Key Takeaways
* **If you want complete interest-free peace of mind:** The **BankAmericard** covers your entire 18-month timeline with its 21-billing-cycle promo period ([source](https://www.bankofamerica.com/credit-cards/products/bankamericard-credit-card/)). However, you cannot use this card to transfer a balance from an existing Bank of America account ([source](https://www.bankofamerica.com/credit-cards/products/bankamericard-credit-card/)).
* **If you want the lowest overall cost:** The **PenFed Gold Visa** saves you $100 upfront on the transfer fee ($150 vs. $250) ([source](https://www.penfed.org/credit-cards/gold-visa)). Even though you will pay interest during the last 3 months, your total out-of-pocket cost is still estimated to be about $74 cheaper than the BankAmericard, provided you pay down the balance consistently.

</td><td valign="top">

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

</td></tr>
</table>

**Comment:** Here the answers disagree. The baseline relies on an outdated 3% BankAmericard transfer fee and recommends the BankAmericard. The agent finds the current 5% fee on the issuer's page, notes how it changed from the 2025 CFPB survey, and correctly identifies the PenFed Gold Visa as the cheaper option.

### Query 3: Is a recommended card the right long-term choice?

> I'm a line cook who eats out on almost every day off, and restaurants and takeout are where most of my money goes, roughly a thousand dollars a month, with a little on groceries. I pay my card off every month and don't want an annual fee. A coworker swears by the U.S. Bank Altitude Go for its 4% back on dining. Would it really be the best card for me over the long run, and what would change that?

**Expected answer:** The Altitude Go should not come out as a clear winner. Its 4% dining rate applies only to the first $2,000 of spending each quarter, so at this level of dining about a third of it earns just 1%, which leaves the card roughly even with an uncapped 3% dining card rather than well ahead of it. The answer should apply the quarterly cap, compare the card with uncapped dining cards and a flat-rate card, explain that eating out more would tip the choice toward an uncapped card, and cite the terms it relies on.

**Where it comes from:**

- Altitude Go dining and grocery rates, the quarterly cap, annual fee and welcome offer: 67.CREDIT dataset, record `usbank-altitude-go`, fields `rewardsRates`, `rewardsNotes`, `annualFee` and `currentBonus`.
- Alternatives: 67.CREDIT records `savorone-cash-rewards`, `wf-autograph` and `wf-active-cash`, plus the [Wells Fargo Autograph page](https://creditcards.wellsfargo.com/autograph-visa-credit-card/) for its uncapped 3% dining rate.
- Foreign transaction fee and APR range: CFPB survey (2025-12-31).

**Agent vs. baseline:**

<table>
<tr><th width="50%">CardWise agent</th><th width="50%">Baseline (no tools, no system prompt)</th></tr>
<tr><td valign="top">

While your coworker’s recommendation of the **U.S. Bank Altitude® Go Visa Signature®** is strong, it may not be the clear-cut winner for you over the long run. 

Because you spend about $1,000 per month ($3,000 per quarter) on dining, you will run directly into the card's quarterly rewards cap. This limitation narrows its lead, making uncapped 3% dining cards nearly identical in value—and potentially better if your spending grows.

---

### Comparison of Key No-Annual-Fee Cards

The table below outlines how the **U.S. Bank Altitude Go** compares to competitive $0 annual fee cards based on your monthly spending of **$1,000 on dining** and **$100 on groceries**.

| Card | Annual Fee | Dining Rewards | Grocery Rewards | Estimated Annual Rewards |
| :--- | :--- | :--- | :--- | :--- |
| **U.S. Bank Altitude® Go** | $0 ([catalog](https://67.credit/card/usbank-altitude-go)) | 4% (4X points) up to $2,000/quarter, then 1% ([catalog](https://67.credit/card/usbank-altitude-go)) | 2% (2X points) ([catalog](https://67.credit/card/usbank-altitude-go)) | **$372 to $384** |
| **Capital One SavorOne Cash Rewards** | $0 ([catalog](https://67.credit/card/savorone-cash-rewards)) | 3% (3X points) uncapped ([catalog](https://67.credit/card/savorone-cash-rewards)) | 1% (1X points) ([catalog](https://67.credit/card/savorone-cash-rewards)) | **$372** |
| **Wells Fargo Autograph℠ Card** | $0 ([catalog](https://67.credit/card/wf-autograph)) | 3% (3X points) uncapped ([issuer](https://creditcards.wellsfargo.com/autograph-visa-credit-card/)) | 1% (1X points) ([catalog](https://67.credit/card/wf-autograph)) | **$372** |
| **Wells Fargo Active Cash® Card** | $0 ([catalog](https://67.credit/card/wf-active-cash)) | 2% uncapped ([catalog](https://67.credit/card/wf-active-cash)) | 2% uncapped ([catalog](https://67.credit/card/wf-active-cash)) | **$264** |

*Note on Point Values:* Point values for all cards above are estimated at $0.01 per point or cash-back dollar ([Altitude Go catalog](https://67.credit/card/usbank-altitude-go), [SavorOne catalog](https://67.credit/card/savorone-cash-rewards), [Autograph catalog](https://67.credit/card/wf-autograph), [Active Cash catalog](https://67.credit/card/wf-active-cash)).

---

### Detailed Calculations

#### 1. U.S. Bank Altitude® Go Visa Signature®
The Altitude Go has a cap on rewards earnings, noted as up to $2,000 per quarter on dining/grocery categories ([catalog](https://67.credit/card/usbank-altitude-go)). 
* **Dining ($3,000 per quarter):** For each quarter, the first $2,000 earns 4% (about $80). The remaining $1,000 exceeds the cap and earns 1% (about $10). 
  * Quarterly Dining Rewards: $80 + $10 = $90
  * Annual Dining Rewards: $90 × 4 quarters = **$360**
* **Groceries ($1,200 per year):** 
  * *Scenario A (If grocery earnings share the $2,000 quarterly cap):* Because your dining spend alone exhausts the $2,000 cap, groceries would earn the base 1% rate. 
    * Annual Grocery Rewards: $1,200 × 1% = **$12**
    * **Total Estimated Rewards:** $360 + $12 = **$372 per year**
  * *Scenario B (If grocery earnings do not share or exceed the cap):* Groceries earn the full 2% rate.
    * Annual Grocery Rewards: $1,200 × 2% = **$24**
    * **Total Estimated Rewards:** $360 + $24 = **$384 per year**

#### 2. Capital One SavorOne Cash Rewards
* **Dining:** $12,000 × 3% = **$360**
* **Groceries:** $1,200 × 1% = **$12**
* **Total Estimated Rewards:** **$372 per year** (no caps apply)

#### 3. Wells Fargo Autograph℠ Card
* **Dining:** $12,000 × 3% = **$360** (using unlimited 3X points on restaurants, ([issuer](https://creditcards.wellsfargo.com/autograph-visa-credit-card/)))
* **Groceries:** $1,200 × 1% = **$12**
* **Total Estimated Rewards:** **$372 per year**

#### 4. Wells Fargo Active Cash® Card
* **Dining & Groceries:** $13,200 total spend × 2% flat rate = **$264 per year**

---

### What Factors Would Change the "Best" Card for You?

While the Altitude Go offers a small edge (up to $12 more per year if groceries earn 2% uncapped), several factors could quickly tip the scales toward an uncapped 3% card like the SavorOne or Autograph:

* **Your Dining Spend Increases:** If you get a raise or dine out more, the Altitude Go's cap will restrict you. For example, if you spend $1,200/month on dining ($3,600/quarter):
  * **Altitude Go** would earn 4% on $2,000 ($80) and 1% on $1,600 ($16) per quarter, totaling $384/year on dining.
  * An **uncapped 3% card** (like SavorOne or Autograph) would earn 3% on $14,400, totaling $432/year on dining—easily beating the Altitude Go.
* **Highly Fluctuating Spend:** If your restaurant spending is seasonal or concentrated in certain months, you may overshoot the $2,000 cap in some quarters and earn only 1% on the surplus, whereas an uncapped 3% card keeps your earnings consistent.
* **Foreign Travel:** If you ever travel or order from international websites, foreign transaction fees matter. The Wells Fargo Active Cash has a 3.0% foreign transaction fee ([CFPB](https://files.consumerfinance.gov/f/documents/cfpb_tccp-data_2025-12-31.xlsx)). Complete foreign transaction fee data was not verified for the other cards, so check their terms if you plan to travel.

### Note on Welcome Offers
If you want to maximize your value in the first year, welcome offers can act as a great tiebreaker:
* **U.S. Bank Altitude® Go:** offers 25,000 points (valued at $250) as of 2026-07-29 ([catalog](https://67.credit/card/usbank-altitude-go)).
* **Capital One SavorOne:** offers $200 cash back as of 2026-07-29 ([catalog](https://67.credit/card/savorone-cash-rewards)).
* **Wells Fargo Autograph℠:** offers 20,000 points (valued at $200) as of 2026-07-29 ([catalog](https://67.credit/card/wf-autograph)).
* **Wells Fargo Active Cash®:** offers $200 cash back as of 2026-07-29 ([catalog](https://67.credit/card/wf-active-cash)).

*Remember: Since you pay your card off in full every month, you can ignore the standard purchase APRs (which historically range from 17.49% to 29.74% across these cards, according to the [December 2025 CFPB survey](https://files.consumerfinance.gov/f/documents/cfpb_tccp-data_2025-12-31.xlsx)).*

---

### Sources
* **U.S. Bank Altitude Go:** [Catalog Terms](https://67.credit/card/usbank-altitude-go) (2026) and [CFPB Survey Data](https://files.consumerfinance.gov/f/documents/cfpb_tccp-data_2025-12-31.xlsx) (2025).
* **Capital One SavorOne:** [Catalog Terms](https://67.credit/card/savorone-cash-rewards) (2026) and [CFPB Survey Data](https://files.consumerfinance.gov/f/documents/cfpb_tccp-data_2025-12-31.xlsx) (2025).
* **Wells Fargo Autograph:** [Catalog Terms](https://67.credit/card/wf-autograph) (2025/2026), [Official Wells Fargo Claim Page](https://creditcards.wellsfargo.com/autograph-visa-credit-card/) (2026), and [CFPB Survey Data](https://files.consumerfinance.gov/f/documents/cfpb_tccp-data_2025-12-31.xlsx) (2025).
* **Wells Fargo Active Cash:** [Catalog Terms](https://67.credit/card/wf-active-cash) (2025/2026) and [CFPB Survey Data](https://files.consumerfinance.gov/f/documents/cfpb_tccp-data_2025-12-31.xlsx) (2025).

</td><td valign="top">

Your coworker gave you excellent advice. For your specific spending habits—heavy dining, virtually no annual fee preference, and paying off your balance monthly—the **U.S. Bank Altitude® Go Visa Signature® Card** is arguably the best card on the market.

Here is a breakdown of why it fits you so well, the math behind it, and what circumstances might make you want to look elsewhere.

---

### The Math: Why the Altitude Go Wins
At $1,000 a month on dining, you are spending $12,000 a year on restaurants, bars, food trucks, and takeout. 

*   **U.S. Bank Altitude Go (4% on dining):** You would earn **$480 a year** in cash back just on your dining spend.
*   **A standard 3% card** (like the Capital One SavorOne or Chase Freedom Unlimited): You would earn **$360 a year**.
*   **A standard 2% card** (like the Wells Fargo Active Cash): You would earn **$240 a year**.

By choosing the Altitude Go over other popular no-fee cards, you are putting an extra $120 to $240 back in your pocket every year. 

**Other Perks of the Card:**
*   **$15 annual streaming credit** if you pay for services like Netflix, Spotify, or Hulu with the card.
*   **No foreign transaction fees**, which is rare for a no-fee card (great if you ever travel abroad).
*   **2% back** on groceries, gas, and streaming (which covers your minor grocery spend).

---

### What about other "top" dining cards?

You might hear people recommend other cards, but your high monthly spend ($1,000) changes the math in favor of the Altitude Go:

1.  **The Citi Custom Cash® Card (5% cash back):** This card gives 5% back on your top spending category, but **only up to $500 of spend per month** (then it drops to 1%). 
    *   *The Math:* You'd get 5% on the first $500 ($25) and 1% on the next $500 ($5). Your monthly total would be $30 ($360/year). **The Altitude Go beats this by $120/year.**
2.  **Capital One SavorOne (3% cash back):** It has no annual fee and is highly recommended, but it only offers 3% on dining. Even though it offers 3% on groceries and entertainment, your massive dining budget means the Altitude Go’s 4% on dining easily beats it out.

---

### What would change this? (Why you might choose a different card)

While the Altitude Go is currently your best bet, here are the factors that would change that decision:

#### 1. You want to travel and are willing to pay an Annual Fee
If you decide you want to start using your credit card points to book flights and hotels instead of getting cash back, your strategy changes. 
*   The **American Express® Gold Card** earns 4x points on dining and US supermarkets. However, it has a **$320 annual fee** (See [Rates and Fees](https://www.americanexpress.com/us/credit-cards/card-application/apply/prospect/terms/amex-gold-card-member/)). 
*   If you can utilize the card's monthly credits (like Uber Cash and dining credits) to offset that fee, and you redeem those points for high-value international flights, the Amex Gold can yield way more than $480 in value. But if you just want simple cash back, stick to the Altitude Go.

#### 2. U.S. Bank's Strict Approval Rules
U.S. Bank is notoriously "inquiry sensitive." They do not like to approve people who have applied for multiple credit cards or loans in the past 6 to 12 months. 
*   *What changes:* If you have opened a couple of credit cards recently, U.S. Bank might deny you. In that case, the **Wells Fargo Autograph℠ Card** (no fee, 3% on dining, travel, transit, gas, phone plans, and streaming) or the **Capital One SavorOne** are much easier to get approved for.

#### 3. Your budget shifts to groceries
If you ever burn out on eating out and start cooking at home, your spending will shift to grocery stores. 
*   *What changes:* The Altitude Go only gets 2% on groceries. If your grocery spend surpasses your dining spend, a card like the **Blue Cash Everyday® Card from American Express** (3% on groceries, online retail, and gas, with no annual fee) would become more valuable.

### The Verdict
Unless you plan to save up points for international flights, **your coworker is 100% right.** 

For a line cook spending $1,000/month on restaurants and takeout who wants a simple, free cash-back setup, the **U.S. Bank Altitude Go is the undisputed king.** Apply for it, use it exclusively for your dining, and enjoy your free $480+ every year.

</td></tr>
</table>

**Comment:** This is where the answers differ most. The baseline treats the 4% dining rate as unlimited, values the card at $480 a year and calls it the clear winner. The agent applies the $2,000 quarterly cap, finds the card earns about $360 on dining and is roughly tied with uncapped 3% cards, and shows how more dining would make an uncapped card the better choice.

### Vague questions

> What's the best credit card for me?

CardWise does not start research when a question lacks the details that decide the answer. For this question, the Lead replies:

> To help me find the best credit card for you, could you please share:
> 1. Your typical **monthly spending** by category (e.g., groceries, dining, gas, travel, utilities)?
> 2. Your preference regarding **annual fees** (e.g., strictly no-fee, or open to annual fees for better rewards)?
> 3. Whether you are interested in **business cards**, or only personal cards?

No Researcher or Writer runs for this turn. When the user answers, the earlier question and the Lead's request are passed back with the reply, and research begins.

