# Resource Allocation Playbook: Flood Relief in Coastal Odisha

## 1. Overview

**Purpose:** Operational guide for NGO/district disaster managers to pre-position flood-relief supplies using ML-driven forecasts.

**Target Area:** 12 coastal districts of Odisha (Balasore, Bhadrak, Kendrapara, Jagatsinghpur, Puri, Ganjam, Khordha, Cuttack, Jajpur, Mayurbhanj, Nayagarh, Keonjhar)

**Forecast Window:** 7–30 days

---

## 2. How to Use the Dashboard

### Step 1: Select District
- Choose your operating district from the dropdown on the left sidebar
- Default: Puri (most flood-prone coastal district)

### Step 2: Set Forecast Horizon
- Slide to 7, 14, or 30 days depending on your logistics lead time
- *7 days*: Tactical deployment decisions
- *30 days*: Strategic pre-positioning

### Step 3: Read the Risk Assessment
| Risk Level | Probability | Action |
|------------|-------------|--------|
| 🔴 HIGH | >30% | Pre-position ≥60% supplies, activate emergency teams |
| 🟡 MODERATE | 10–30% | Review inventory, update logistics plan, alert volunteers |
| 🟢 LOW | <10% | Standard monitoring, continue regular operations |

### Step 4: Check Resource Table
The dashboard shows SPHERE-standard resource quantities:
- **Medical Kits**: 1 kit per 1,000 affected people
- **ORS Packets**: 200 packets per 1,000 affected people
- **Tarpaulin Sheets**: 1 sheet per family (5 people)
- **Drinking Water**: 15L/person/day (30-day supply)
- **Dry Rations**: 400g/person/day (30-day supply)

### Step 5: Review Timeline
- Historical flood events (red crosses)
- Forecasted demand (orange dashed line)
- 80% confidence interval (shaded area)
- Use the lower bound for minimum stock, upper bound for full prepositioning

---

## 3. Operational Workflow

### Pre-Monsoon (March–May)
- Run forecast for all 12 districts
- Identify top-3 highest-risk districts
- Pre-position 30% of anticipated supplies in strategic warehouses
- Train community volunteers on early warning

### Monsoon (June–September)
- Run weekly forecast updates
- When risk >30%: Mobilize mobile medical units
- When risk >50%: Activate all emergency protocols, request NDRF coordination
- Track actual rainfall vs forecast

### Post-Monsoon (October–November)
- Run damage assessment
- Replenish supplies based on actual consumption
- Update forecast model with new data

### Winter (December–February)
- Low flood risk period
- Conduct inventory audit
- Train team on dashboard usage
- Plan next year's budget

---

## 4. Logistics Guidelines

### Warehouse Pre-Positioning

| District | Strategic Location | Capacity (MT) |
|----------|-------------------|---------------|
| Puri | District HQ | 50 |
| Cuttack | Central Warehouse | 100 |
| Balasore | Northern Hub | 75 |
| Ganjam | Southern Hub | 75 |

### Supply Chain Timeline
- **Day 0**: Forecast triggers alert
- **Day 1–2**: Confirm with IMD rainfall forecast
- **Day 3–5**: Mobilize supplies from warehouse
- **Day 5–7**: Deliver to distribution points
- **Day 7–30**: Distribute to affected population

### Per-Kit Contents
| Kit Type | Contents | Weight | Cost (₹) |
|----------|----------|--------|-----------|
| Medical Kit | Bandages, antiseptic, paracetamol, ORS, gloves | 5 kg | 1,200 |
| Family Kit | Tarpaulin, water purifier, dry rations, utensils | 25 kg | 3,500 |

---

## 5. Understanding Model Uncertainty

The dashboard shows forecast uncertainty as:
- **80% Confidence Interval**: There is an 80% chance actual demand falls within this range
- **Wider interval = Higher uncertainty** (typically in distant months and post-monsoon)
- **Decision rule**: Budget using lower bound, but have plans to scale to upper bound

### When to Trust the Forecast
- ✅ District has >5 flood events in training data
- ✅ Current month is June–September (monsoon)
- ✅ Model confidence interval is <50% of the mean

### When to Be Cautious
- ⚠️ New district with limited historical data
- ⚠️ Extreme weather events (cyclones) not well captured
- ⚠️ Forecast beyond 30 days

---

## 6. Data Sources & Updates

| Data | Source | Update Frequency |
|------|--------|-----------------|
| Flood Events | India Flood Inventory (IIT Delhi/IMD) | Annual |
| Flood Severity Index | IIT Delhi (DFSI) | Annual |
| Population | Census 2011 | Decennial |
| District Boundaries | LGD/Government of India | As notified |
| Humanitarian Standards | SPHERE Handbook | Every 5 years |

---

## 7. Cost-Benefit Estimate

| Metric | Without Forecast | With Forecast |
|--------|-----------------|---------------|
| Response time | 72–96 hours | 24–48 hours |
| Relief delivery cost | ₹100/person | ₹70/person (~30% savings) |
| People served (₹10L budget) | 5,000 | 15,000 |
| Supply wastage | 25–40% | 5–15% |

---

## 8. Contact & Support

- **Dashboard Issues**: data@example.org
- **IMD Forecast**: https://mausam.imd.gov.in
- **NDMA Guidelines**: https://ndma.gov.in
- **SPHERE Standards**: https://spherestandards.org

*Last Updated: June 2026*
