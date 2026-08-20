from typing import List, Dict, Tuple, Optional
from app.v3.reasoning.schemas import CalculationStep


def calculate_roa(
    net_income: float,
    beg_assets: float,
    end_assets: float,
) -> Tuple[float, List[CalculationStep]]:
    avg_assets = (beg_assets + end_assets) / 2.0
    if avg_assets == 0:
        raise ZeroDivisionError("Average Total Assets is zero.")
    result = net_income / avg_assets

    steps = [
        CalculationStep(
            step_number=1,
            description="Calculate Average Total Assets from Beginning and Ending Assets",
            expression=f"({beg_assets:,.2f} + {end_assets:,.2f}) / 2",
            intermediate_value=avg_assets,
        ),
        CalculationStep(
            step_number=2,
            description="Divide Net Income by Average Total Assets",
            expression=f"{net_income:,.2f} / {avg_assets:,.2f}",
            intermediate_value=result,
        ),
    ]
    return result, steps


def calculate_roe(
    net_income: float,
    beg_equity: float,
    end_equity: float,
) -> Tuple[float, List[CalculationStep]]:
    avg_equity = (beg_equity + end_equity) / 2.0
    if avg_equity == 0:
        raise ZeroDivisionError("Average Shareholders' Equity is zero.")
    result = net_income / avg_equity

    steps = [
        CalculationStep(
            step_number=1,
            description="Calculate Average Shareholders' Equity",
            expression=f"({beg_equity:,.2f} + {end_equity:,.2f}) / 2",
            intermediate_value=avg_equity,
        ),
        CalculationStep(
            step_number=2,
            description="Divide Net Income by Average Shareholders' Equity",
            expression=f"{net_income:,.2f} / {avg_equity:,.2f}",
            intermediate_value=result,
        ),
    ]
    return result, steps


def calculate_dpo(
    accounts_payable: float,
    cogs: float,
    days: float = 365.0,
    prev_accounts_payable: Optional[float] = None,
) -> Tuple[float, List[CalculationStep]]:
    if cogs == 0:
        raise ZeroDivisionError("Cost of Goods Sold (COGS) is zero.")

    if prev_accounts_payable is not None:
        avg_ap = (accounts_payable + prev_accounts_payable) / 2.0
        ap_desc = f"Average Accounts Payable: ({prev_accounts_payable:,.2f} + {accounts_payable:,.2f}) / 2"
    else:
        avg_ap = accounts_payable
        ap_desc = f"Accounts Payable: {accounts_payable:,.2f}"

    result = (avg_ap / cogs) * days

    steps = [
        CalculationStep(
            step_number=1,
            description=ap_desc,
            expression=f"{avg_ap:,.2f}",
            intermediate_value=avg_ap,
        ),
        CalculationStep(
            step_number=2,
            description=f"Calculate DPO = (Average AP / COGS) * {days:.0f} Days",
            expression=f"({avg_ap:,.2f} / {cogs:,.2f}) * {days:.0f}",
            intermediate_value=result,
        ),
    ]
    return result, steps


def calculate_inventory_turnover(
    cogs: float,
    curr_inventory: float,
    prev_inventory: Optional[float] = None,
) -> Tuple[float, List[CalculationStep]]:
    if prev_inventory is not None:
        avg_inv = (curr_inventory + prev_inventory) / 2.0
        inv_desc = f"Average Inventory: ({prev_inventory:,.2f} + {curr_inventory:,.2f}) / 2"
    else:
        avg_inv = curr_inventory
        inv_desc = f"Ending Inventory: {curr_inventory:,.2f}"

    if avg_inv == 0:
        raise ZeroDivisionError("Average Inventory is zero.")

    result = cogs / avg_inv

    steps = [
        CalculationStep(
            step_number=1,
            description=inv_desc,
            expression=f"{avg_inv:,.2f}",
            intermediate_value=avg_inv,
        ),
        CalculationStep(
            step_number=2,
            description="Calculate Inventory Turnover = COGS / Average Inventory",
            expression=f"{cogs:,.2f} / {avg_inv:,.2f}",
            intermediate_value=result,
        ),
    ]
    return result, steps


def calculate_operating_margin(
    operating_income: float,
    revenue: float,
) -> Tuple[float, List[CalculationStep]]:
    if revenue == 0:
        raise ZeroDivisionError("Revenue is zero.")
    result = operating_income / revenue

    steps = [
        CalculationStep(
            step_number=1,
            description="Divide Operating Income by Revenue",
            expression=f"{operating_income:,.2f} / {revenue:,.2f}",
            intermediate_value=result,
        ),
    ]
    return result, steps


def calculate_capex_ratio(
    capex: float,
    revenue: float,
) -> Tuple[float, List[CalculationStep]]:
    if revenue == 0:
        raise ZeroDivisionError("Revenue is zero.")
    result = capex / revenue

    steps = [
        CalculationStep(
            step_number=1,
            description="Divide CAPEX by Revenue",
            expression=f"{capex:,.2f} / {revenue:,.2f}",
            intermediate_value=result,
        ),
    ]
    return result, steps


def calculate_yoy_growth(
    curr_value: float,
    prev_value: float,
) -> Tuple[float, List[CalculationStep]]:
    if prev_value == 0:
        raise ZeroDivisionError("Previous period value is zero.")
    result = (curr_value - prev_value) / prev_value

    steps = [
        CalculationStep(
            step_number=1,
            description="Calculate Change = (Current Value - Previous Value)",
            expression=f"{curr_value:,.2f} - {prev_value:,.2f}",
            intermediate_value=curr_value - prev_value,
        ),
        CalculationStep(
            step_number=2,
            description="Divide Change by Previous Value",
            expression=f"({curr_value:,.2f} - {prev_value:,.2f}) / {prev_value:,.2f}",
            intermediate_value=result,
        ),
    ]
    return result, steps


def calculate_multi_year_average(
    values: List[float],
) -> Tuple[float, List[CalculationStep]]:
    if not values:
        raise ValueError("Cannot calculate average of empty list.")
    total = sum(values)
    count = len(values)
    result = total / count

    steps = [
        CalculationStep(
            step_number=1,
            description=f"Sum values across {count} periods",
            expression=" + ".join([f"{v:,.2f}" for v in values]),
            intermediate_value=total,
        ),
        CalculationStep(
            step_number=2,
            description=f"Divide Total by {count} periods",
            expression=f"{total:,.2f} / {count}",
            intermediate_value=result,
        ),
    ]
    return result, steps


def calculate_fcf_conversion(
    fcf: float,
    net_income: float,
) -> Tuple[float, List[CalculationStep]]:
    if net_income == 0:
        raise ZeroDivisionError("Net Income is zero.")
    result = fcf / net_income

    steps = [
        CalculationStep(
            step_number=1,
            description="Divide Free Cash Flow by Net Income",
            expression=f"{fcf:,.2f} / {net_income:,.2f}",
            intermediate_value=result,
        ),
    ]
    return result, steps
