import numpy as np
import logging
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TechnoEconomicModel:
    """
    A robust Techno-Economic and Agricultural Yield Model for Agrivoltaic Systems.
    
    Mathematical Framework:
    This model computes the financial viability of an agrivoltaic project by 
    combining dual revenue streams:
    1. Electricity Revenue: E_yield * P_electricity * A_pv
    2. Agricultural Revenue: Crop_yield * P_crop * A_agri * ETR_effect
    
    It outputs rigorous metrics such as:
    - Net Present Value (NPV)
    - Return on Investment (ROI)
    - Total Revenue over Time ($/ha/yr)
    - Levelized Cost of Energy (LCOE)
    """
    
    def __init__(self, 
                 discount_rate: float = 0.05,
                 system_lifetime: float = 20.0,
                 electricity_price: float = 0.12, # $/kWh
                 crop_price_per_kg: float = 1.50):
        self.discount_rate = discount_rate
        self.system_lifetime = system_lifetime
        self.electricity_price = electricity_price
        self.crop_price_per_kg = crop_price_per_kg
        logger.info("Techno-Economic Model initialized.")
        
    def evaluate_project_viability(self, 
                                   area_hectares: float,
                                   pv_coverage_ratio: float,
                                   pce: float,
                                   etr: float,
                                   base_crop_yield_kg_per_ha: float = 8000.0,
                                   capex_per_kw: float = 1200.0,
                                   opex_per_kw_yr: float = 15.0) -> Dict[str, Any]:
        """
        Evaluates the ROI, NPV, and combined revenue for a specified system size.
        
        Parameters:
        - area_hectares: Total farm area
        - pv_coverage_ratio: Fraction of area covered by OPV
        - pce: Power Conversion Efficiency of the OPV module
        - etr: Energy Transfer Ratio (Light passing to crops)
        """
        # Convert Area
        total_area_m2 = area_hectares * 10000.0
        pv_area_m2 = total_area_m2 * pv_coverage_ratio
        agri_area_m2 = total_area_m2 # Crops grow under and between panels
        
        # 1. PV Energy Yield Calculation
        # Assuming 1700 kWh/m2/yr as baseline solar irradiance
        annual_energy_kwh = 1700.0 * pv_area_m2 * pce
        system_capacity_kw = (1000.0 * pv_area_m2 * pce) / 1000.0 # 1000 W/m2 standard test condition irradiance
        
        electricity_revenue_yr = annual_energy_kwh * self.electricity_price
        
        # 2. Agricultural Yield Calculation
        # ETR represents the fraction of PAR (Photosynthetically Active Radiation) surviving
        # For shade-tolerant crops, yield is slightly boosted, for sun-loving, it scales somewhat linearly with ETR
        # We assume a moderate response: Yield = Base_Yield * (1 - PV_Coverage + PV_Coverage*ETR)
        effective_light_ratio = (1.0 - pv_coverage_ratio) + (pv_coverage_ratio * etr)
        
        # Apply a mild shading benefit to water retention, assuming 1.1x boost modifier on purely light-driven yields
        crop_yield_kg_yr = base_crop_yield_kg_per_ha * area_hectares * effective_light_ratio * 1.1
        agricultural_revenue_yr = crop_yield_kg_yr * self.crop_price_per_kg
        
        # 3. Financials
        total_revenue_yr = electricity_revenue_yr + agricultural_revenue_yr
        
        capex = system_capacity_kw * capex_per_kw
        opex_yr = system_capacity_kw * opex_per_kw_yr
        
        net_cash_flow_yr = total_revenue_yr - opex_yr
        
        # Calculate NPV
        cash_flows = [net_cash_flow_yr] * int(self.system_lifetime)
        npv = -capex + sum([cf / (1 + self.discount_rate)**t for t, cf in enumerate(cash_flows, start=1)])
        
        # Calculate ROI
        total_net_profit = (net_cash_flow_yr * self.system_lifetime) - capex
        roi = (total_net_profit / capex) * 100.0 if capex > 0 else 0
        
        # Payback period
        payback_period = capex / net_cash_flow_yr if net_cash_flow_yr > 0 else float('inf')
        
        # Levelized Cost of Energy over purely PV side
        total_discounted_cost = capex + sum([opex_yr / (1 + self.discount_rate)**t for t in range(1, int(self.system_lifetime)+1)])
        total_discounted_energy = sum([annual_energy_kwh / (1 + self.discount_rate)**t for t in range(1, int(self.system_lifetime)+1)])
        lcoe = total_discounted_cost / total_discounted_energy if total_discounted_energy > 0 else 0
        
        return {
            'area_hectares': area_hectares,
            'pv_coverage_ratio': pv_coverage_ratio,
            'capex_usd': capex,
            'electricity_revenue_yr_usd': electricity_revenue_yr,
            'agricultural_revenue_yr_usd': agricultural_revenue_yr,
            'total_revenue_yr_usd_per_ha': total_revenue_yr / area_hectares,
            'npv_usd': npv,
            'roi_percent': roi,
            'payback_period_years': payback_period,
            'lcoe_usd_per_kwh': lcoe
        }
