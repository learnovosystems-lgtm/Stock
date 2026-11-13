"""NSE liquid universe with cap bucket and sector tags.

Universe is a curated, liquid snapshot to reduce survivorship and illiquidity
issues inherent in free historical feeds. Extend via data/universe_extra.csv.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ati.config import DATA_DIR
from ati.models.opportunity import CapBucket


@dataclass(frozen=True)
class Instrument:
    symbol: str  # Yahoo suffix .NS already attached for equities
    name: str
    sector: str
    cap: CapBucket
    sector_index: str | None = None


# Working Yahoo sector index proxies for India
SECTOR_INDEX = {
    "IT": "^CNXIT",
    "AUTO": "^CNXAUTO",
    "PHARMA": "^CNXPHARMA",
    "FMCG": "^CNXFMCG",
    "METAL": "^CNXMETAL",
    "ENERGY": "^CNXENERGY",
    "REALTY": "^CNXREALTY",
    "PSU_BANK": "^CNXPSUBANK",
    "PRIVATE_BANK": "^NSEBANK",
    "FINANCIALS": "^CNXFIN",
    "INFRA": "^CNXINFRA",
    "MEDIA": "^CNXMEDIA",
}


# Curated liquid NSE names (Yahoo symbols)
_SEED: list[tuple[str, str, str, CapBucket]] = [
    # Large caps
    ("RELIANCE.NS", "Reliance Industries", "ENERGY", CapBucket.LARGE),
    ("TCS.NS", "Tata Consultancy", "IT", CapBucket.LARGE),
    ("HDFCBANK.NS", "HDFC Bank", "PRIVATE_BANK", CapBucket.LARGE),
    ("ICICIBANK.NS", "ICICI Bank", "PRIVATE_BANK", CapBucket.LARGE),
    ("INFY.NS", "Infosys", "IT", CapBucket.LARGE),
    ("ITC.NS", "ITC", "FMCG", CapBucket.LARGE),
    ("SBIN.NS", "State Bank of India", "PSU_BANK", CapBucket.LARGE),
    ("BHARTIARTL.NS", "Bharti Airtel", "TELECOM", CapBucket.LARGE),
    ("LT.NS", "Larsen & Toubro", "INFRA", CapBucket.LARGE),
    ("AXISBANK.NS", "Axis Bank", "PRIVATE_BANK", CapBucket.LARGE),
    ("KOTAKBANK.NS", "Kotak Mahindra Bank", "PRIVATE_BANK", CapBucket.LARGE),
    ("HINDUNILVR.NS", "Hindustan Unilever", "FMCG", CapBucket.LARGE),
    ("BAJFINANCE.NS", "Bajaj Finance", "FINANCIALS", CapBucket.LARGE),
    ("MARUTI.NS", "Maruti Suzuki", "AUTO", CapBucket.LARGE),
    ("SUNPHARMA.NS", "Sun Pharma", "PHARMA", CapBucket.LARGE),
    ("TITAN.NS", "Titan", "CONSUMER", CapBucket.LARGE),
    ("ASIANPAINT.NS", "Asian Paints", "CONSUMER", CapBucket.LARGE),
    ("ULTRACEMCO.NS", "UltraTech Cement", "CEMENT", CapBucket.LARGE),
    ("NTPC.NS", "NTPC", "ENERGY", CapBucket.LARGE),
    ("POWERGRID.NS", "Power Grid", "ENERGY", CapBucket.LARGE),
    ("ONGC.NS", "ONGC", "ENERGY", CapBucket.LARGE),
    ("TMPV.NS", "Tata Motors Passenger", "AUTO", CapBucket.LARGE),
    ("TATASTEEL.NS", "Tata Steel", "METAL", CapBucket.LARGE),
    ("JSWSTEEL.NS", "JSW Steel", "METAL", CapBucket.LARGE),
    ("WIPRO.NS", "Wipro", "IT", CapBucket.LARGE),
    ("HCLTECH.NS", "HCL Tech", "IT", CapBucket.LARGE),
    ("TECHM.NS", "Tech Mahindra", "IT", CapBucket.LARGE),
    ("ADANIENT.NS", "Adani Enterprises", "CONGLOMERATE", CapBucket.LARGE),
    ("ADANIPORTS.NS", "Adani Ports", "INFRA", CapBucket.LARGE),
    ("COALINDIA.NS", "Coal India", "ENERGY", CapBucket.LARGE),
    ("BAJAJFINSV.NS", "Bajaj Finserv", "FINANCIALS", CapBucket.LARGE),
    ("NESTLEIND.NS", "Nestle India", "FMCG", CapBucket.LARGE),
    ("M&M.NS", "Mahindra & Mahindra", "AUTO", CapBucket.LARGE),
    ("CIPLA.NS", "Cipla", "PHARMA", CapBucket.LARGE),
    ("DRREDDY.NS", "Dr Reddy's", "PHARMA", CapBucket.LARGE),
    ("EICHERMOT.NS", "Eicher Motors", "AUTO", CapBucket.LARGE),
    ("HEROMOTOCO.NS", "Hero MotoCorp", "AUTO", CapBucket.LARGE),
    ("INDUSINDBK.NS", "IndusInd Bank", "PRIVATE_BANK", CapBucket.LARGE),
    ("BPCL.NS", "BPCL", "ENERGY", CapBucket.LARGE),
    ("GRASIM.NS", "Grasim", "CEMENT", CapBucket.LARGE),
    # Mid caps
    ("PERSISTENT.NS", "Persistent Systems", "IT", CapBucket.MID),
    ("COFORGE.NS", "Coforge", "IT", CapBucket.MID),
    ("DIXON.NS", "Dixon Technologies", "CONSUMER", CapBucket.MID),
    ("POLYCAB.NS", "Polycab", "INDUSTRIAL", CapBucket.MID),
    ("CUMMINSIND.NS", "Cummins India", "INDUSTRIAL", CapBucket.MID),
    ("VOLTAS.NS", "Voltas", "CONSUMER", CapBucket.MID),
    ("TVSMOTOR.NS", "TVS Motor", "AUTO", CapBucket.MID),
    ("ASHOKLEY.NS", "Ashok Leyland", "AUTO", CapBucket.MID),
    ("FEDERALBNK.NS", "Federal Bank", "PRIVATE_BANK", CapBucket.MID),
    ("AUROPHARMA.NS", "Aurobindo Pharma", "PHARMA", CapBucket.MID),
    ("BIOCON.NS", "Biocon", "PHARMA", CapBucket.MID),
    ("GODREJPROP.NS", "Godrej Properties", "REALTY", CapBucket.MID),
    ("DLF.NS", "DLF", "REALTY", CapBucket.MID),
    ("PIIND.NS", "PI Industries", "CHEMICALS", CapBucket.MID),
    ("AARTIIND.NS", "Aarti Industries", "CHEMICALS", CapBucket.MID),
    ("INDIGO.NS", "InterGlobe Aviation", "AVIATION", CapBucket.MID),
    ("IRCTC.NS", "IRCTC", "CONSUMER", CapBucket.MID),
    ("PFC.NS", "Power Finance Corp", "FINANCIALS", CapBucket.MID),
    ("RECLTD.NS", "REC Ltd", "FINANCIALS", CapBucket.MID),
    ("HDFCAMC.NS", "HDFC AMC", "FINANCIALS", CapBucket.MID),
    # Small / smaller mid liquid names
    ("KAYNES.NS", "Kaynes Technology", "INDUSTRIAL", CapBucket.SMALL),
    ("BSE.NS", "BSE Ltd", "FINANCIALS", CapBucket.SMALL),
    ("ANGELONE.NS", "Angel One", "FINANCIALS", CapBucket.SMALL),
    ("CAMS.NS", "CAMS", "FINANCIALS", CapBucket.SMALL),
    ("KPITTECH.NS", "KPIT Technologies", "IT", CapBucket.SMALL),
    ("SONACOMS.NS", "Sona BLW", "AUTO", CapBucket.SMALL),
    ("ZYDUSLIFE.NS", "Zydus Lifesciences", "PHARMA", CapBucket.MID),
    ("LUPIN.NS", "Lupin", "PHARMA", CapBucket.MID),
    ("HAVELLS.NS", "Havells", "CONSUMER", CapBucket.MID),
    ("BERGEPAINT.NS", "Berger Paints", "CONSUMER", CapBucket.MID),
    # Expanded liquid names for Phase-2 research sample
    ("BEL.NS", "Bharat Electronics", "INDUSTRIAL", CapBucket.LARGE),
    ("HAL.NS", "Hindustan Aeronautics", "INDUSTRIAL", CapBucket.LARGE),
    ("SIEMENS.NS", "Siemens", "INDUSTRIAL", CapBucket.LARGE),
    ("ABB.NS", "ABB India", "INDUSTRIAL", CapBucket.MID),
    ("CGPOWER.NS", "CG Power", "INDUSTRIAL", CapBucket.MID),
    ("APOLLOHOSP.NS", "Apollo Hospitals", "PHARMA", CapBucket.LARGE),
    ("DIVISLAB.NS", "Divi's Labs", "PHARMA", CapBucket.LARGE),
    ("TORNTPHARM.NS", "Torrent Pharma", "PHARMA", CapBucket.MID),
    ("BRITANNIA.NS", "Britannia", "FMCG", CapBucket.LARGE),
    ("DABUR.NS", "Dabur", "FMCG", CapBucket.MID),
    ("MARICO.NS", "Marico", "FMCG", CapBucket.MID),
    ("GODREJCP.NS", "Godrej Consumer", "FMCG", CapBucket.MID),
    ("TATACONSUM.NS", "Tata Consumer", "FMCG", CapBucket.MID),
    ("BAJAJ-AUTO.NS", "Bajaj Auto", "AUTO", CapBucket.LARGE),
    ("BOSCHLTD.NS", "Bosch", "AUTO", CapBucket.MID),
    ("MRF.NS", "MRF", "AUTO", CapBucket.MID),
    ("BALKRISIND.NS", "Balkrishna Ind", "AUTO", CapBucket.MID),
    ("HDFCLIFE.NS", "HDFC Life", "FINANCIALS", CapBucket.LARGE),
    ("SBILIFE.NS", "SBI Life", "FINANCIALS", CapBucket.LARGE),
    ("ICICIGI.NS", "ICICI Lombard", "FINANCIALS", CapBucket.MID),
    ("CHOLAFIN.NS", "Cholamandalam", "FINANCIALS", CapBucket.MID),
    ("MUTHOOTFIN.NS", "Muthoot Finance", "FINANCIALS", CapBucket.MID),
    ("BANKBARODA.NS", "Bank of Baroda", "PSU_BANK", CapBucket.LARGE),
    ("CANBK.NS", "Canara Bank", "PSU_BANK", CapBucket.MID),
    ("PNB.NS", "Punjab National Bank", "PSU_BANK", CapBucket.MID),
    ("VEDL.NS", "Vedanta", "METAL", CapBucket.LARGE),
    ("HINDALCO.NS", "Hindalco", "METAL", CapBucket.LARGE),
    ("NMDC.NS", "NMDC", "METAL", CapBucket.MID),
    ("SAIL.NS", "SAIL", "METAL", CapBucket.MID),
    ("IOC.NS", "Indian Oil", "ENERGY", CapBucket.LARGE),
    ("GAIL.NS", "GAIL", "ENERGY", CapBucket.MID),
    ("ADANIGREEN.NS", "Adani Green", "ENERGY", CapBucket.MID),
    ("TATAPOWER.NS", "Tata Power", "ENERGY", CapBucket.MID),
    ("MPHASIS.NS", "Mphasis", "IT", CapBucket.MID),
    ("OFSS.NS", "Oracle Financial", "IT", CapBucket.MID),
    ("NAUKRI.NS", "Info Edge", "IT", CapBucket.MID),
    ("TRENT.NS", "Trent", "CONSUMER", CapBucket.LARGE),
    ("DMART.NS", "Avenue Supermarts", "CONSUMER", CapBucket.LARGE),
    ("PAGEIND.NS", "Page Industries", "CONSUMER", CapBucket.MID),
    ("JUBLFOOD.NS", "Jubilant FoodWorks", "CONSUMER", CapBucket.MID),
    ("OBEROIRLTY.NS", "Oberoi Realty", "REALTY", CapBucket.MID),
    ("PRESTIGE.NS", "Prestige Estates", "REALTY", CapBucket.MID),
    ("LODHA.NS", "Macrotech Developers", "REALTY", CapBucket.MID),
    ("AMBUJACEM.NS", "Ambuja Cements", "CEMENT", CapBucket.LARGE),
    ("SHREECEM.NS", "Shree Cement", "CEMENT", CapBucket.MID),
    ("DALBHARAT.NS", "Dalmia Bharat", "CEMENT", CapBucket.MID),
    ("SRF.NS", "SRF", "CHEMICALS", CapBucket.MID),
    ("PIDILITIND.NS", "Pidilite", "CHEMICALS", CapBucket.LARGE),
    ("UPL.NS", "UPL", "CHEMICALS", CapBucket.MID),
]


def _to_instrument(row: tuple[str, str, str, CapBucket]) -> Instrument:
    symbol, name, sector, cap = row
    return Instrument(
        symbol=symbol,
        name=name,
        sector=sector,
        cap=cap,
        sector_index=SECTOR_INDEX.get(sector),
    )


def load_universe(extra_csv: Path | None = None) -> list[Instrument]:
    instruments = [_to_instrument(r) for r in _SEED]
    path = extra_csv or (DATA_DIR / "universe_extra.csv")
    if path.exists():
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            instruments.append(
                Instrument(
                    symbol=str(r["symbol"]),
                    name=str(r.get("name", r["symbol"])),
                    sector=str(r.get("sector", "UNKNOWN")),
                    cap=CapBucket(str(r.get("cap", "MID")).upper()),
                    sector_index=SECTOR_INDEX.get(str(r.get("sector", "")).upper()),
                )
            )
    # de-dupe by symbol
    seen: set[str] = set()
    out: list[Instrument] = []
    for inst in instruments:
        if inst.symbol not in seen:
            seen.add(inst.symbol)
            out.append(inst)
    return out


def symbols_by_cap(universe: list[Instrument] | None = None) -> dict[CapBucket, list[str]]:
    universe = universe or load_universe()
    out: dict[CapBucket, list[str]] = {c: [] for c in CapBucket}
    for inst in universe:
        out[inst.cap].append(inst.symbol)
    return out
