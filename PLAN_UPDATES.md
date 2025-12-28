# Plan Updates: Iterative Testing and Reference to Findings Document

## Key Updates to Implementation Plan

### 1. Reference to findings_and_planning.md
**CRITICAL:** The plan should explicitly reference `findings_and_planning.md` throughout execution. This document contains:
- Detailed analysis results from Step 1
- Specific trial examples and their characteristics  
- Root cause analysis of failures
- Evidence-based rationale for each fix
- Success metrics and targets
- Key insights and discoveries

### 2. Iterative Testing Process
**CRITICAL WORKFLOW:** After EACH implementation step:
1. **Run error analysis suite:**
   ```bash
   python -m swap_correction.analyze_errors --all-trials --summary
   ```
2. **Compare metrics to baseline** (documented in `findings_and_planning.md`):
   - Segment detection rate (baseline: 0-10%, target: >90%)
   - Number of problematic trials (baseline: 11, target: <3)
   - Average segment count per problematic trial (baseline: 1-21, target: <2)
   - Error rate for problematic trials (baseline: 11-41%, target: <1%)
3. **Verify no regressions:**
   - All 14 perfect trials must remain perfect (0% error rate, 0 segments)
   - Check specific trials mentioned in `findings_and_planning.md`:
     - Global swap trial: `2024.11.13_00-48-15_Sussex_e2hex`
     - Multi-segment trial: `2024.11.13_00-19-29_Sussex_e2hex`
4. **Document results:**
   - Record metrics before/after in `findings_and_planning.md`
   - Note any issues or unexpected behaviors
   - Only proceed to next step if current step shows improvement or no regression

### 3. Updates to Each Phase

Each phase should now include:
- **Reference:** Links to relevant sections in `findings_and_planning.md`
- **Validation:** Explicit testing steps after each implementation
- **Documentation:** Requirement to update `findings_and_planning.md` with results

### 4. Specific Trial References

The plan should reference specific trials from `findings_and_planning.md`:
- Global swap trial: `2024.11.13_00-48-15_Sussex_e2hex` (3287 frames, cross-sign match=0.617)
- Multi-segment trial: `2024.11.13_00-19-29_Sussex_e2hex` (18 segments, max 1509 frames)
- Perfect trials: 14 trials that must remain perfect

