# Exclusion List Comparison: User vs Data-Driven

## Summary

| List | Combinations | Volume Excluded | Portfolio Coverage | Logic |
|------|--------------|-----------------|-------------------|-------|
| **User** | 14 | €1,013M (3.03%) | 96.97% | Business logic + ref map |
| **Data-Driven** | 16 | €1,127M (3.37%) | 96.63% | Performance metrics |
| **Difference** | -2 | -€114M (-0.34 pp) | +0.34 pp | User covers more |

---

## Discrepancy Analysis

### **Combinations User EXCLUDES but Data Says KEEP** (2 combinations)

#### 1. **20B2-TRP** - User: ❌ EXCLUDE | Data: ✅ KEEP
**Data Performance**: GOOD
- W1-W8 WAPE: 19.4% - 24.0% (avg 21.3%)
- Volume: €90.3M (0.27% of portfolio)
- Data weeks: 193 (excellent history)
- CV: 0.87 (moderate volatility)

**Reference Map**: "No LP inputs presently"

**User's Rationale** (my interpretation):
- Ref map explicitly states "No LP inputs presently"
- If we exclude from ML, we have no LP fallback → risky
- BUT: User wants to exclude anyway, possibly because:
  - Entity 20B2 has data quality concerns (TRR has 137% WAPE)
  - Prefer to exclude entire entity for consistency
  - Business decision to treat 20B2 as Tier 2

**My Counter**: Performance is GOOD (21% WAPE), but I defer to business judgment

**Recommendation**: ✅ **Accept user's exclusion** (business logic overrides data)

---

#### 2. **T055-TRP** - User: ❌ EXCLUDE | Data: ✅ KEEP
**Data Performance**: EXCELLENT
- W1-W8 WAPE: 16.2% - 21.8% (avg 19.6%)
- Volume: €25.9M (0.08% of portfolio)
- Data weeks: 187 (excellent history)
- CV: 0.77 (low volatility)
- **WAPE improves over horizons** (W8=16.2% better than W1=20.1%)

**Reference Map**: "Usually TRR, with small amount of payables"

**User's Rationale**:
- Ref map explicitly says "**small amount of payables**"
- TRP is not the primary flow for T055
- Focus ML on primary flows (TRR for T055)
- TRP volume is minimal (€26M vs TRR €500M for T055)

**My Counter**: Performance is excellent, but ref map is clear about business reality

**Recommendation**: ✅ **Accept user's exclusion** (aligns with ref map guidance)

---

### **Combinations Data EXCLUDES but User KEEPS** (4 combinations)

#### 3. **4B9-TRP** - User: ✅ KEEP | Data: ❌ EXCLUDE
**Data Performance**: POOR
- W1-W8 WAPE: 51.4% - 58.6% (avg 55.2%)
- Volume: €40.1M (0.12% of portfolio)
- Data weeks: 190 (good history)
- CV: 1.59 (high volatility)

**Reference Map**: "LP updates sometimes missing; liquidity mix unspecified"

**Data's Rationale**: Poor WAPE (55%), exclude
**User's Rationale**:
- Volume is €40M (not trivial)
- Need coverage for this entity-liq combination
- Willing to accept higher WAPE for coverage

**My Analysis**:
- Excluding saves 0.12% of portfolio
- WAPE is 2.5x worse than target (55% vs 20% acceptable)
- But LP updates sometimes missing (per ref map) → ML may be better than LP!

**Recommendation**: ⚠️ **Borderline - defer to user, but flag as high-error combination**

---

#### 4. **V756-TRP** - User: ✅ KEEP | Data: ❌ EXCLUDE
**Data Performance**: POOR
- W1-W8 WAPE: 51.7% - 55.9% (avg 54.4%)
- Volume: €147.0M (0.44% of portfolio - **SIGNIFICANT!**)
- Data weeks: 193 (excellent history)
- CV: 1.29 (high volatility)

**Reference Map**: No special comments (both TRR and TRP active)

**Data's Rationale**: Poor WAPE (54%), exclude
**User's Rationale**:
- **€147M is material!** (0.44% of portfolio)
- Cannot exclude this much volume
- Need forecasts even if not perfect

**My Analysis**:
- This is the **LARGEST** of the 4 discrepancies
- Excluding €147M may be too aggressive
- User is right: 0.44% of portfolio is material
- Better to keep and flag as "high uncertainty" forecast

**Recommendation**: ✅ **Accept user's decision - volume too material to exclude**

---

#### 5. **057-TRP** - User: ✅ KEEP | Data: ❌ EXCLUDE
**Data Performance**: POOR
- W1-W8 WAPE: 46.3% - 50.6% (avg 48.9%)
- Volume: €22.9M (0.07% of portfolio)
- Data weeks: 160 (moderate history)
- CV: 1.28 (high volatility)

**Reference Map**: No special comments (both TRR and TRP active)

**Data's Rationale**: Poor WAPE (49%), small volume, exclude
**User's Rationale**:
- Need coverage for all entity-liq combinations
- Volume is €23M (not trivial in absolute terms)

**My Analysis**:
- WAPE is ~49% (2x worse than good performers)
- Volume is small (0.07% of portfolio)
- But: Only 160 weeks vs 196 for most others → may improve with more data

**Recommendation**: ⚠️ **Weak keep - marginal combination, but defer to user**

---

#### 6. **V508-TRP** - User: ✅ KEEP | Data: ❌ EXCLUDE
**Data Performance**: POOR
- W1-W8 WAPE: 45.1% - 47.2% (avg 46.1%)
- Volume: €20.6M (0.06% of portfolio)
- Data weeks: 196 (excellent history)
- CV: 0.97 (moderate volatility)

**Reference Map**: No special comments (both TRR and TRP active)

**Data's Rationale**: Poor WAPE (46%), small volume, exclude
**User's Rationale**:
- Need coverage
- WAPE is 46% (not as bad as others like 4B9-TRP at 55%)

**My Analysis**:
- WAPE is ~46% (better than 4B9-TRP, V756-TRP, 057-TRP)
- Volume is minimal (0.06% of portfolio)
- Has full 196 weeks of data → well-tested

**Recommendation**: ⚠️ **Weak keep - marginal impact either way**

---

## Core Philosophy Difference

**Data-Driven Approach (Mine)**:
- Prioritize **WAPE quality** over coverage
- Exclude combinations with WAPE > 40% and low volume (<1% portfolio)
- **Goal**: Optimize portfolio WAPE by removing worst performers

**Business-Driven Approach (User's)**:
- Prioritize **coverage** - need forecasts for all material flows
- Use **ref map business logic** (e.g., "small amount of payables" → exclude)
- Willing to accept higher WAPE on some combinations for completeness
- **Goal**: Cover 97% of portfolio with ML, even if some forecasts are imperfect

---

## Recommendation: HYBRID APPROACH ✅

**Proposed Final List** (14 combinations - User's list):

| Entity | Liq Group | Reason | Volume | W1 WAPE |
|--------|-----------|--------|--------|---------|
| **82J** | TRR | Not in data | - | - |
| **82J** | TRP | Not in data | - | - |
| **25A4** | TRR | Not in data | - | - |
| **25A4** | TRP | Not in data | - | - |
| **11G5** | TRR | 81% WAPE, not expected | €6.9M | 81% |
| **14C1** | TRR | 126% WAPE, not expected | €176M | 126% |
| **17C7** | TRR | 101% WAPE, data quality | €80.8M | 101% |
| **20B2** | TRR | 137% WAPE, insufficient data | €2.0M | 137% |
| **20B2** | TRP | ✅ Business decision (ref map) | €90.3M | 19.4% |
| **T056** | TRR | 292% WAPE, not expected | €237M | 292% |
| **V002** | TRR | 54% WAPE, not expected | €69.5M | 54% |
| **V265** | TRP | 47% WAPE, ref map logic | €105M | 47% |
| **86W** | TRP | 45% WAPE, ref map logic | €219M | 45% |
| **T055** | TRP | ✅ Ref map: "small payables" | €25.9M | 20.1% |

**Impact**:
- Excluded volume: €1,013M (3.03%)
- Portfolio coverage: **96.97%** ✅
- ML forecasting: 20 combinations

---

## Special Treatment for 4 Borderline Combinations

User wants to KEEP but have poor WAPE:
1. **4B9-TRP**: 55% WAPE, €40M
2. **V756-TRP**: 54% WAPE, **€147M** (material!)
3. **057-TRP**: 49% WAPE, €23M
4. **V508-TRP**: 46% WAPE, €21M

**Recommendation**:
- ✅ Keep in ML forecasting (per user's request)
- 🚨 Flag as **"High Uncertainty Forecasts"** in production
- 📊 Report with wider confidence intervals
- 🔍 Monthly monitoring - if WAPE doesn't improve after new features, revisit

---

## FINAL RECOMMENDATION

**Accept User's List** ✅ (14 exclusions)

**Rationale**:
1. **Business logic is valid**: Ref map explicitly guides some exclusions
2. **Coverage matters**: 97% portfolio coverage is important
3. **Material volumes**: V756-TRP (€147M) too large to exclude
4. **Consistency**: 20B2 entire entity excluded (both TRR and TRP)
5. **Data is guidance, not mandate**: User knows business better

**Action Items**:
1. Implement user's 14 exclusions as Tier 2 (LP passthrough)
2. Flag 4 borderline combinations (4B9-TRP, V756-TRP, 057-TRP, V508-TRP) for monitoring
3. Build ML models for remaining 20 combinations
4. Report "high uncertainty" tags for borderline combinations

---

**Bottom Line**:
User's list is **more conservative on coverage** (keeps marginal performers), **more aligned with business reality** (ref map), and achieves **96.97% coverage**. Data-driven list optimizes WAPE but sacrifices coverage.

**I recommend: User's list wins!** ✅
