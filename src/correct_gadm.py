# %%
import pandas as pd
import difflib
from configparser import ConfigParser


def string_diff(
    s: str, gadm_spa: pd.DataFrame, munic2comarca: pd.DataFrame, munic_version: str
) -> str | None:

    assert (
        len(set(gadm_spa["NAME_2"].unique()) - set(munic2comarca["provincia"].unique()))
        == 0
    ), "Provinces names should be equal!"

    prov = munic2comarca.query(f"{munic_version} == @s").iloc[0]["provincia"]
    munic_gadm = gadm_spa.query("NAME_2 == @prov")["NAME_4"].unique()
    m = difflib.get_close_matches(s, munic_gadm, n=1, cutoff=0.9)

    if len(m) == 1:
        return m[0]
    else:
        return None


def find_na(x: str) -> bool:
    if isinstance(x, str):
        return x.find("n.a.") != -1
    elif x == None:
        return True
    else:
        return False


def correct(gadm: pd.DataFrame, config: ConfigParser) -> None:
    """Modify in place GADM level 3 names relative to Spain."""

    gadm_spa = gadm.query("NAME_0 == 'Spain'").copy()
    munic2comarca = pd.read_csv(config["paths"]["munic2comarca"])

    munic_version = "municipio_v2"
    munic2comarca["municipio_gadm"] = munic2comarca[munic_version].apply(
        string_diff, args=(gadm_spa, munic2comarca, munic_version)
    )

    munic2comarca["id"] = (
        munic2comarca["provincia"] + "_" + munic2comarca["municipio_gadm"]
    )
    gadm_spa["id"] = gadm_spa["NAME_2"] + "_" + gadm_spa["NAME_4"]
    gadm_spa_comarca = pd.merge(
        gadm_spa[["id", "UID", "NAME_3", "NAME_4"]],
        munic2comarca[["id", "municipio_gadm", "comarca"]],
        on="id",
        how="left",
    ).drop(columns=["id"])

    assert gadm_spa_comarca[
        "UID"
    ].is_unique, (
        "UID should be unique! Try to increase word similarity cutoff parameter."
    )

    # Available gadm_spa-level3 name but missing "comarca" for given gadm_spa-level4 municipality
    idx_na_comarca = pd.isna(gadm_spa_comarca["comarca"]) & ~gadm_spa_comarca[
        "NAME_3"
    ].apply(find_na)

    print(
        f"{idx_na_comarca.sum()} missing 'comarca' names from the reference table are replaced from available GADM names"
    )

    gadm_spa_comarca.loc[~pd.isna(gadm_spa_comarca["comarca"]), "NAME_3"] = (
        gadm_spa_comarca[~pd.isna(gadm_spa_comarca["comarca"])].loc[
            ~idx_na_comarca, "comarca"
        ]
    )

    gadm_spa_comarca.set_index("UID", inplace=True)
    gadm.set_index("UID", inplace=True)
    gadm.update(gadm_spa_comarca["NAME_3"])
    gadm.reset_index(inplace=True)

    perc_isna_comarca = (
        (gadm.query("NAME_0 == 'Spain'")["NAME_3"].apply(find_na)).sum()
        / len(gadm_spa)
        * 100
    )

    print(
        f"There is still {perc_isna_comarca:.2f}% of missing Spanish municipalities that miss a 'comarca' name."
    )
