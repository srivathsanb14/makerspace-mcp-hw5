"""HW5 Q4: MCP server exposing five analytical tools (3 from HW4 + 2 new).

Run:  python server.py            (stdio transport, for Inspector / any MCP client)
Original (HW4):  discover_thermal_model, discover_cop_law, daily_heating_energy
New (HW5):       part_load_check, failure_cost_ledger

Invalid input: every tool raises ValueError with a plain message; the MCP SDK returns it as a
tool result with isError=true (pydantic range checks on the typed fields give the same shape).
"""
import math
from typing import Annotated, Literal
from pydantic import Field
from mcp.server.fastmcp import FastMCP
import hw4_calcs as H

mcp = FastMCP(
    "makerspace-analytics",
    instructions=(
        "Five analytical tools in two groups. HVAC group (from HW4, synthetic house data): discover_thermal_model, "
        "discover_cop_law, daily_heating_energy. 3D-printing group (HW5, from maker interviews): part_load_check, "
        "failure_cost_ledger. Never invent inputs the user did not give (load, material, rates). Always show the "
        "'warnings' and 'assumptions' fields. Results are estimates; a person decides whether to act."),
)


# ==================== HVAC group (HW4) ====================
# ------------------------------------------------------------------ original tool 1
@mcp.tool()
def discover_thermal_model(
    threshold: Annotated[float, Field(ge=0.01, le=1, description="Sparsity threshold, K/h per std of each term (0.01-1)")] = 0.1,
    window_pts: Annotated[int, Field(ge=6, le=72, description="Integration window in 5-min samples (6-72)")] = 24,
) -> dict:
    """[HVAC group, HW4] Discover the equation governing indoor (room) temperature from the 1-week thermostat log
    (weak-form sparse regression). Returns equation (K/h), coefficients, R2, tau (h), UA/C (1/h),
    heater gain (K/h at full duty), solar gain, and warnings. Synthetic data; verify before real use."""
    return H.calc_discover_thermal_model(threshold, window_pts)


# ------------------------------------------------------------------ original tool 2
@mcp.tool()
def discover_cop_law(
    include_all_candidates: Annotated[bool, Field(description="Also return all 8 ranked candidate formulas")] = False,
) -> dict:
    """[HVAC group, HW4] Search 8 candidate formulas for heat-pump COP vs outdoor/supply temperature (log covers outdoor
    -15..15 C, supply 30..55 C), ranked by BIC. Returns best formula, params, R2, RMSE, Carnot fraction."""
    r = H.calc_discover_cop_law()
    if not include_all_candidates:
        r = {k: v for k, v in r.items() if k != "ranked"} | {"runner_up": r["ranked"][1]}
    return r


# ------------------------------------------------------------------ original tool 3
@mcp.tool()
def daily_heating_energy(
    T_out_C: Annotated[float, Field(ge=-30, le=20, description="Daily mean outdoor temperature, degC (-30..20, must be < setpoint)")],
    T_set_C: Annotated[float, Field(ge=15, le=26, description="Indoor setpoint, degC (15-26)")] = 21.0,
    T_supply_C: Annotated[float, Field(ge=30, le=60, description="Heat-pump supply-water temperature, degC (30-60)")] = 40.0,
    price_per_kWh: Annotated[float, Field(ge=0, le=5, description="Electricity price, $/kWh")] = 0.17,
) -> dict:
    """[HVAC group, HW4] Steady-state daily heat demand (kWh/day), COP, electricity (kWh/day) and cost ($/day) using the
    discovered UA and Carnot fraction. Warns when extrapolating beyond the COP test range."""
    return H.calc_daily_heating_energy(T_out_C, T_set_C, T_supply_C, price_per_kWh=price_per_kWh)


# ==================== 3D-printing group (HW5) ====================
# ------------------------------------------------------------------ NEW tool 4
# Typical datasheet tensile strengths for FDM filament along the layers (MPa). Approximate: check your filament.
MATERIAL_STRENGTH_MPA = {"PLA": 50.0, "PETG": 45.0, "ABS": 35.0, "NYLON": 60.0}
ORIENTATION_FACTOR = {"flat": 1.0, "upright": 0.5}       # load across layers is weaker (layer adhesion)
RESUME_FACTOR = {"no": 1.0, "yes_no_reheat": 0.52, "yes_reheated": 0.52 * 1.47}  # HW4: interruption -48%, reheating +47%


@mcp.tool()
def part_load_check(
    material: Annotated[Literal["PLA", "PETG", "ABS", "NYLON"], Field(description="Filament material")],
    load_N: Annotated[float, Field(gt=0, le=5000, description="Applied point load, newtons (0-5000)")],
    span_mm: Annotated[float, Field(ge=5, le=500, description="Beam span or cantilever length, mm (5-500)")],
    width_mm: Annotated[float, Field(ge=1, le=100, description="Cross-section width b, mm (1-100)")],
    height_mm: Annotated[float, Field(ge=1, le=100, description="Cross-section height h in the bending direction, mm (1-100)")],
    support: Annotated[Literal["cantilever", "simply_supported"], Field(description="Load case: end load on a cantilever, or centre load on a simply supported beam")] = "cantilever",
    orientation: Annotated[Literal["flat", "upright"], Field(description="Print orientation: 'flat' loads along the layers, 'upright' loads across them")] = "flat",
    resumed_after_pause: Annotated[Literal["no", "yes_no_reheat", "yes_reheated"], Field(description="Was the print interrupted and resumed, and was the layer reheated first?")] = "no",
    required_safety_factor: Annotated[float, Field(ge=1, le=10, description="Safety factor the part must meet (1-10)")] = 2.0,
) -> dict:
    """[3D-printing group, HW5] Quick function check for a printed part (interviews: makers want proof that a part, especially a
    rescued one, still does its job). Textbook bending stress in a rectangular beam against a typical material
    strength, knocked down for print orientation and for a resumed (paused) print. Returns stress, allowable
    stress, safety factor and pass/fail. An ESTIMATE from approximate datasheet values, not a stress test."""
    M = load_N * span_mm if support == "cantilever" else load_N * span_mm / 4          # N*mm
    stress = 6 * M / (width_mm * height_mm ** 2)                                         # MPa
    base = MATERIAL_STRENGTH_MPA[material]
    allowable = base * ORIENTATION_FACTOR[orientation] * RESUME_FACTOR[resumed_after_pause]
    sf = allowable / stress
    warnings = []
    if span_mm < 2 * height_mm:
        warnings.append("Span under 2x height: not a slender beam, bending theory is unreliable.")
    if sf < 1:
        warnings.append("Predicted to fail under this load.")
    elif sf < required_safety_factor:
        warnings.append(f"Below the required safety factor of {required_safety_factor}.")
    if resumed_after_pause == "yes_no_reheat":
        warnings.append("Resumed without reheating: strength cut by 48% (HW4, Sinha & Meisel 2018). Reheat or reprint.")
    if resumed_after_pause != "no":
        warnings.append("Resume knock-down is an average from one study; verify with a physical test of a rescued part.")
    return {
        "bending_stress_MPa": round(stress, 2), "base_strength_MPa": base, "allowable_stress_MPa": round(allowable, 2),
        "knockdown_orientation": ORIENTATION_FACTOR[orientation], "knockdown_resume": round(RESUME_FACTOR[resumed_after_pause], 3),
        "safety_factor": round(sf, 2), "passes_required_safety_factor": bool(sf >= required_safety_factor),
        "warnings": warnings,
        "assumptions": "Rectangular solid section; static load; typical room-temperature XY tensile strength; no stress concentrations or infill reduction. Not a substitute for a physical load test.",
    }


# ------------------------------------------------------------------ NEW tool 5
@mcp.tool()
def failure_cost_ledger(
    n_printers: Annotated[int, Field(ge=1, le=200, description="Printers in the fleet (1-200)")],
    jobs_per_printer_per_week: Annotated[float, Field(gt=0, le=50, description="Jobs started per printer per week (0-50)")] = 8.0,
    avg_job_hours: Annotated[float, Field(gt=0, le=72, description="Mean planned job length, hours (0-72)")] = 6.0,
    failure_rate: Annotated[float, Field(ge=0, le=0.6, description="Fraction of jobs that fail (0-0.6). 0.138 = Formlabs benchmark")] = 0.138,
    mean_fraction_done_at_failure: Annotated[float, Field(gt=0, le=1, description="How far through a job it fails on average (0-1)")] = 0.5,
    avg_material_g: Annotated[float, Field(gt=0, le=5000, description="Material per completed job, grams")] = 120.0,
    material_cost_per_kg: Annotated[float, Field(ge=0, le=500, description="Material cost, $/kg")] = 20.0,
    printer_hour_value: Annotated[float, Field(ge=0, le=100, description="Value of one printer-hour, $/h (opportunity cost)")] = 1.5,
    recoverable_fraction: Annotated[float, Field(ge=0, le=1, description="What-if: share of failed jobs a recover-and-resume system could save (0-1)")] = 0.0,
    resume_overhead_hours: Annotated[float, Field(ge=0, le=24, description="Extra hours a rescued job still costs (pause, reheat)")] = 0.5,
) -> dict:
    """[3D-printing group, HW5] Make failure cost visible (HW4 gap 12): weekly wasted printer-hours, grams and dollars from failed
    jobs, plus an optional what-if of what a recover-and-resume capability would save. Planning arithmetic
    on user-supplied rates, not a measurement; replace defaults with the space's own logs."""
    jobs = n_printers * jobs_per_printer_per_week
    failed = jobs * failure_rate
    hrs_lost = failed * avg_job_hours * mean_fraction_done_at_failure
    kg_lost = failed * avg_material_g * mean_fraction_done_at_failure / 1000
    mat_cost = kg_lost * material_cost_per_kg
    time_cost = hrs_lost * printer_hour_value
    total = mat_cost + time_cost
    saved_jobs = failed * recoverable_fraction
    saved_hrs = saved_jobs * (avg_job_hours * mean_fraction_done_at_failure - resume_overhead_hours)
    saved_kg = saved_jobs * avg_material_g * mean_fraction_done_at_failure / 1000
    warnings = []
    if jobs * avg_job_hours > n_printers * 168:
        warnings.append("Planned load exceeds 168 h/week per printer: inputs are not physically feasible.")
    if resume_overhead_hours > avg_job_hours * mean_fraction_done_at_failure and recoverable_fraction > 0:
        warnings.append("Resume overhead exceeds the work saved: resuming loses time versus restarting.")
    return {
        "jobs_per_week": round(jobs, 1), "failed_jobs_per_week": round(failed, 2),
        "printer_hours_wasted_per_week": round(hrs_lost, 1),
        "material_wasted_kg_per_week": round(kg_lost, 2),
        "material_cost_per_week": round(mat_cost, 2), "printer_time_cost_per_week": round(time_cost, 2),
        "total_failure_cost_per_week": round(total, 2),
        "fleet_utilization_lost_pct": round(100 * hrs_lost / (n_printers * 168), 2),
        "what_if_resume": {"jobs_saved_per_week": round(saved_jobs, 2),
                           "printer_hours_saved_per_week": round(max(saved_hrs, 0), 1),
                           "material_saved_kg_per_week": round(saved_kg, 2),
                           "dollars_saved_per_week": round(max(saved_hrs, 0) * printer_hour_value + saved_kg * material_cost_per_kg, 2)},
        "warnings": warnings,
    }


if __name__ == "__main__":
    mcp.run()
