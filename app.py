import math
import re
from datetime import datetime
from html import escape
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

API_BASE = "https://api.ptr.zanz2.dev/api/nations"
TIMEOUT = 30


# ----------------------------- API / data ---------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def get_json(url: str) -> dict | list:
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def month_key(value: str | None) -> int | None:
    """Convert YYYY-MM game month into a sortable integer."""
    if not value:
        return None
    try:
        year, month = value.split("-")
        return int(year) * 12 + int(month)
    except (ValueError, AttributeError):
        return None


def provisions_map(constitution: dict) -> dict[str, Any]:
    return {
        p.get("system_parameter_key"): p.get("current_value")
        for p in constitution.get("provisions", [])
        if p.get("system_parameter_key")
    }


def active_joint_lists(joint_lists: list, poll_month: str) -> dict[int, dict]:
    """Return joint lists and their consenting members at the poll's game month."""
    when = month_key(poll_month)
    result = {}
    for item in joint_lists:
        activated = month_key(item.get("first_activated_at_game_month"))
        dissolved = month_key(item.get("dissolved_at_game_month"))
        if activated is None or when is None or activated > when:
            continue
        if dissolved is not None and dissolved <= when:
            continue

        members = []
        for member in item.get("members", []):
            joined = month_key(member.get("joined_at_game_month"))
            left = month_key(member.get("left_at_game_month"))
            if member.get("member_status") != "consented":
                continue
            if joined is not None and joined > when:
                continue
            if left is not None and left <= when:
                continue
            members.append(member)

        if len(members) >= 2:
            result[int(item["id"])] = {**item, "eligible_members": members}
    return result


def get_provision_value(provisions: dict, key: str, default=None):
    value = provisions.get(key, default)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def get_discrete_provision(constitution: dict, key: str, default: str = "") -> str:
    for provision in constitution.get("provisions", []):
        if provision.get("system_parameter_key") == key:
            value_name = provision.get("current_value_name")
            if value_name:
                return str(value_name)
            current_value = provision.get("current_value")
            for value in provision.get("values", []):
                if str(value.get("id")) == str(current_value):
                    resolved_name = value.get("name")
                    if resolved_name:
                        return str(resolved_name)
            return str(current_value or default)
    return default


def derive_electoral_mode(constitution: dict, poll: dict) -> str:
    apportionment = get_discrete_provision(
        constitution, "regional_apportionment",
        str(poll.get("allocation_method") or "")
    ).lower()
    threshold_scope = get_discrete_provision(
        constitution, "regional_threshold_scope", ""
    ).lower()

    if "national" in apportionment:
        return "National pool"
    if "whole" in threshold_scope or "nation" in threshold_scope:
        return "Regional allocation, national threshold"
    return "District threshold"


def derive_allocation_algorithm(constitution: dict) -> str:
    algorithm = get_discrete_provision(
        constitution, "seat_allocation_algorithm", "Sainte-Lague"
    ).lower()
    if "d'hondt" in algorithm or "dhondt" in algorithm or algorithm.strip() == "9":
        return "D'Hondt"
    if "hare" in algorithm or "largest remainder" in algorithm or algorithm.strip() == "11":
        return "Hare-largest remainder"
    return "Sainte-Lague"




def government_records(payload: dict | list) -> list[dict]:
    """Accept either a single government object or a list/API wrapper."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("items", "governments", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [payload] if payload.get("id") is not None else []


def latest_government(payload: dict | list) -> dict | None:
    """Choose the government with the latest formation month."""
    records = government_records(payload)
    if not records:
        return None
    return max(
        records,
        key=lambda item: (month_key(item.get("formed_at")) or -1,
                          str(item.get("created_at") or ""),
                          int(item.get("id") or 0)),
    )


def current_government_party_ids(government: dict | None) -> set[int]:
    """Get parties with at least one currently serving government member."""
    if not government:
        return set()
    party_ids = set()
    for member in government.get("members", []):
        if member.get("party_id") is None:
            continue
        if any(member.get(field) is not None for field in (
            "resigned_at", "dismissed_by_party_id", "impeached_bill_id",
            "abolished_at_game_month"
        )):
            continue
        if member.get("party_is_active") is False:
            continue
        party_ids.add(int(member["party_id"]))
    return party_ids


def add_largest_party_probability(frame: pd.DataFrame,
                                  scope_results: list[dict[int, int]]) -> pd.DataFrame:
    """Add outright-largest and tied-largest probabilities for each party."""
    if frame.empty or not scope_results:
        frame["Chance largest party"] = "0.0%"
        frame["Chance tied for largest"] = "0.0%"
        return frame
    party_ids = frame["Party ID"].astype(int).tolist()
    sole_wins = {pid: 0 for pid in party_ids}
    ties = {pid: 0 for pid in party_ids}
    for result in scope_results:
        largest = max((int(result.get(pid, 0)) for pid in party_ids), default=0)
        leaders = [
            pid for pid in party_ids
            if int(result.get(pid, 0)) == largest
        ]
        if len(leaders) == 1:
            sole_wins[leaders[0]] += 1
        else:
            for pid in leaders:
                ties[pid] += 1
    total = len(scope_results)
    frame["Chance largest party"] = [
        f"{100 * sole_wins[pid] / total:.1f}%" for pid in party_ids
    ]
    frame["Chance tied for largest"] = [
        f"{100 * ties[pid] / total:.1f}%" for pid in party_ids
    ]
    return frame


def build_government_opposition_summary(
    national_runs: list[dict[int, int]], government_party_ids: set[int],
    total_seats: int
) -> pd.DataFrame:
    """Summarise government and opposition seat totals across simulations."""
    if not national_runs or not government_party_ids:
        return pd.DataFrame()
    majority = int(total_seats) // 2 + 1
    gov_values = np.asarray([
        sum(int(seats) for pid, seats in result.items() if int(pid) in government_party_ids)
        for result in national_runs
    ], dtype=int)
    opposition_values = np.asarray([
        sum(int(seats) for pid, seats in result.items() if int(pid) not in government_party_ids)
        for result in national_runs
    ], dtype=int)
    rows = []
    for label, values in (("Government", gov_values), ("Opposition", opposition_values)):
        rows.append({
            "Side": label,
            "Median seats": int(np.floor(np.median(values) + 0.5)),
            "90% CI range": f"{np.percentile(values, 5):.0f}–{np.percentile(values, 95):.0f}",
            "Chance of majority": f"{100 * np.mean(values >= majority):.1f}%",
        })
    return pd.DataFrame(rows)

# -------------------------- Monte Carlo maths ----------------------------

def sample_vote_shares(parties: list[dict], rng: np.random.Generator) -> dict[int, float]:
    """
    Sample each party from N(poll%, (MOE/1.96)^2), reject negative draws,
    then normalise to 100. This preserves the supplied marginal 95% MoE
    before the compositional normalisation step.
    """
    ids, means, sds = [], [], []
    for party in parties:
        value = party.get("support_pct")
        if value is None:
            continue
        moe = party.get("moe")
        ids.append(int(party["party_id"]))
        means.append(max(0.0, float(value)))
        sds.append(max(0.0, float(moe or 0.0) / 1.96))

    if not ids:
        return {}

    means = np.asarray(means, dtype=float)
    sds = np.asarray(sds, dtype=float)
    draws = rng.normal(means, sds)
    # Rejection sampling gives a true lower-truncated normal rather than clipping
    # a negative result to zero and creating an artificial pile-up at zero.
    for _ in range(20):
        bad = draws < 0
        if not bad.any():
            break
        draws[bad] = rng.normal(means[bad], sds[bad])
    draws = np.maximum(draws, 0.0)
    total = draws.sum()
    if total <= 0:
        draws = means.copy()
        total = draws.sum()
    if total <= 0:
        draws = np.ones(len(ids))
        total = draws.sum()
    draws = draws / total * 100.0
    return {party_id: float(vote) for party_id, vote in zip(ids, draws)}


def group_key(party: dict, joint_lists: dict[int, dict]) -> str:
    """Map parties to a qualifying list; otherwise they compete individually."""
    for field in ("electoral_list_id", "partial_list_id"):
        list_id = party.get(field)
        if list_id is not None and int(list_id) in joint_lists:
            return f"list:{int(list_id)}"
    return f"party:{int(party['party_id'])}"


def allocate_sainte_lague(votes: dict[str, float], seats: int,
                           eligible: set[str] | None = None) -> dict[str, int]:
    """Allocate seats using Sainte-Laguë odd divisors 1, 3, 5, ..."""
    allocation = {key: 0 for key in votes}
    if seats <= 0:
        return allocation

    candidates = [
        key for key, value in votes.items()
        if value > 0 and (eligible is None or key in eligible)
    ]
    for _ in range(seats):
        if not candidates:
            break
        winner = max(candidates, key=lambda key: votes[key] / (2 * allocation[key] + 1))
        allocation[winner] += 1
    return allocation


def allocate_dhondt(votes: dict[str, float], seats: int,
                    eligible: set[str] | None = None) -> dict[str, int]:
    """Allocate seats using D'Hondt divisors 1, 2, 3, ..."""
    allocation = {key: 0 for key in votes}
    if seats <= 0:
        return allocation

    candidates = [
        key for key, value in votes.items()
        if value > 0 and (eligible is None or key in eligible)
    ]
    for _ in range(seats):
        if not candidates:
            break
        winner = max(candidates, key=lambda key: votes[key] / (allocation[key] + 1))
        allocation[winner] += 1
    return allocation


def allocate_hare_largest_remainder(votes: dict[str, float], seats: int,
                                    eligible: set[str] | None = None) -> dict[str, int]:
    """Allocate seats using the Hare quota and largest remainders."""
    allocation = {key: 0 for key in votes}
    if seats <= 0:
        return allocation

    candidates = [
        key for key, value in votes.items()
        if value > 0 and (eligible is None or key in eligible)
    ]
    if not candidates:
        return allocation

    total_votes = sum(votes[key] for key in candidates)
    if total_votes <= 0:
        return allocation

    quota = total_votes / seats
    exact = {key: votes[key] / quota for key in candidates}
    for key in candidates:
        allocation[key] = math.floor(exact[key])

    remainder = seats - sum(allocation.values())
    if remainder <= 0:
        return allocation

    order = sorted(
        candidates,
        key=lambda key: (exact[key] - math.floor(exact[key]), key),
        reverse=True,
    )
    for key in order[:remainder]:
        allocation[key] += 1
    return allocation


def allocate_seats(votes: dict[str, float], seats: int,
                   eligible: set[str] | None,
                   algorithm: str) -> dict[str, int]:
    if algorithm == "D'Hondt":
        return allocate_dhondt(votes, seats, eligible)
    if algorithm == "Hare-largest remainder":
        return allocate_hare_largest_remainder(votes, seats, eligible)
    return allocate_sainte_lague(votes, seats, eligible)


def threshold_eligible(votes: dict[str, float], parties: list[dict],
                       joint_lists: dict[int, dict], threshold: float,
                       list_threshold: float | None = None) -> set[str]:
    """
    Apply threshold to standalone parties and pooled joint lists.
    The threshold is measured against the total valid vote in the relevant pool.
    """
    eligible = set()
    list_keys = {group_key(p, joint_lists) for p in parties}
    for key in list_keys:
        vote = votes.get(key, 0.0)
        cutoff = list_threshold if key.startswith("list:") and list_threshold is not None else threshold
        if vote >= cutoff:
            eligible.add(key)
    return eligible


def distribute_group_seats(group_seats: dict[str, int],
                           party_votes: dict[int, float],
                           parties: list[dict] | dict[int, dict],
                           joint_lists: dict[int, dict]) -> dict[int, int]:
    """Split joint-list seats by membership ratios, renormalised to local runners.

    ``parties`` may be either the API's list of party rows or the ID-keyed
    metadata mapping used for district-specific affiliations.
    """
    party_rows = list(parties.values()) if isinstance(parties, dict) else parties
    output = {int(p["party_id"]): 0 for p in party_rows}
    for group, seats in group_seats.items():
        if seats <= 0:
            continue
        runners = [p for p in party_rows if group_key(p, joint_lists) == group]
        if not runners:
            continue
        if group.startswith("list:"):
            list_id = int(group.split(":", 1)[1])
            ratio_by_party = {
                int(member["party_id"]): max(0.0, float(member.get("ratio_pct") or 0.0))
                for member in joint_lists.get(list_id, {}).get("eligible_members", [])
                if member.get("party_id") is not None
            }
            # Only parties actually running in this district participate.
            # Renormalise their membership ratios to 100% for this district.
            weights = {
                int(p["party_id"]): ratio_by_party.get(int(p["party_id"]), 0.0)
                for p in runners
            }
        else:
            weights = {int(runners[0]["party_id"]): 1.0}

        total = sum(weights.values())
        if total <= 0:
            # Never substitute polling strength for missing membership ratios.
            continue

        exact = {pid: seats * weight / total for pid, weight in weights.items()}
        floors = {pid: math.floor(value) for pid, value in exact.items()}
        remainder = seats - sum(floors.values())
        order = sorted(exact, key=lambda pid: (exact[pid] - floors[pid], -pid), reverse=True)
        for pid, count in floors.items():
            output[pid] += count
        for pid in order[:remainder]:
            output[pid] += 1
    return output


def poll_parties_by_id(poll: dict) -> dict[int, dict]:
    return {int(p["party_id"]): p for p in poll.get("parties", [])}


def party_groups(sampled: dict[int, float], party_meta: dict[int, dict],
                 joint_lists: dict[int, dict]) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for pid, vote in sampled.items():
        party = party_meta.get(pid)
        if not party:
            continue
        key = group_key(party, joint_lists)
        grouped[key] = grouped.get(key, 0.0) + vote
    return grouped


def simulate_once(poll: dict, constitution: dict, joint_lists: dict[int, dict],
                  mode: str, threshold: float, list_threshold: float | None,
                  allocation_algorithm: str,
                  rng: np.random.Generator) -> tuple[dict[int, int], dict[str, dict[int, int]]]:
    parties = poll.get("parties", [])
    meta = poll_parties_by_id(poll)
    party_ids = list(meta)
    national_result = {pid: 0 for pid in party_ids}
    region_result: dict[str, dict[int, int]] = {}

    if mode == "National pool":
        sampled = sample_vote_shares(parties, rng)
        grouped = party_groups(sampled, meta, joint_lists)
        eligible = threshold_eligible(grouped, parties, joint_lists, threshold, list_threshold)
        group_seats = allocate_seats(
            grouped, int(poll.get("total_seats", 0)), eligible, allocation_algorithm
        )
        national_result = distribute_group_seats(group_seats, sampled, parties, joint_lists)
        region_result["National"] = national_result.copy()
        return national_result, region_result

    # District allocation: use the API's point-in-time constitutional
    # apportionment, falling back to poll district seat fields only if absent.
    constitution_apportionment = {
        int(x["electoral_district_id"]): int(x["seat_count"])
        for x in constitution.get("apportionment", [])
        if x.get("electoral_district_id") is not None and x.get("seat_count") is not None
    }
    district_rows = poll.get("districts", [])
    national_sample = sample_vote_shares(parties, rng)
    national_grouped = party_groups(national_sample, meta, joint_lists)
    national_eligible = threshold_eligible(
        national_grouped, parties, joint_lists, threshold, list_threshold
    )
    # Members of a national joint list may stand independently in particular
    # districts. Preserve their individual national threshold qualification as
    # well as the pooled list qualification, so a locally standalone party is
    # not excluded merely because it belongs to a list elsewhere.
    for pid, vote in national_sample.items():
        if vote >= threshold:
            national_eligible.add(f"party:{pid}")

    for district in district_rows:
        district_id = int(district.get("district_id", 0))
        district_name = district.get("name", f"District {district_id}")
        district_party_rows = district.get("parties", [])
        # District data has the same party IDs and local MoEs.
        district_sample = sample_vote_shares(district_party_rows, rng)
        local_rows_by_id = {int(row["party_id"]): row for row in district_party_rows}
        district_meta = {}
        for pid in district_sample:
            local_row = local_rows_by_id.get(pid, {})
            party_info = {**meta.get(pid, {}), **local_row}
            # A party's national list membership must not leak into districts
            # where it is standing independently. Treat district-level list
            # fields as authoritative, including when omitted from the row.
            for field in ("electoral_list_id", "partial_list_id"):
                party_info[field] = local_row.get(field)
            district_meta[pid] = party_info

        grouped = party_groups(district_sample, district_meta, joint_lists)

        if mode == "District threshold":
            eligible = threshold_eligible(
                grouped, list(district_meta.values()), joint_lists, threshold, list_threshold
            )
        else:  # Regional allocation, national threshold
            eligible = national_eligible

        seats = constitution_apportionment.get(district_id)
        if seats is None:
            # Last-resort fallback only; prefer the constitution's apportionment.
            seats = sum(int(x.get("projected_seats", 0) or 0) for x in district_party_rows)

        group_seats = allocate_seats(grouped, seats, eligible, allocation_algorithm)
        local_party_seats = distribute_group_seats(
            group_seats, district_sample, district_meta, joint_lists
        )
        region_result[district_name] = local_party_seats
        for pid, count in local_party_seats.items():
            national_result[pid] = national_result.get(pid, 0) + count

    return national_result, region_result


# ----------------------------- Presentation -------------------------------

def build_summary(scope_results: list[dict[int, int]], parties: list[dict],
                  scope_name: str) -> pd.DataFrame:
    """Build a scope-specific table using that scope's own poll estimates."""
    rows = []
    for party in parties:
        pid = int(party["party_id"])
        values = np.asarray([result.get(pid, 0) for result in scope_results], dtype=float)
        party_name = (party.get("party_name") or party.get("name")
                      or party.get("abbreviation") or str(pid))
        rows.append({
            "Party ID": pid,
            "Party": party_name,
            "Party name": party_name,
            "Poll%": party.get("support_pct"),
            "MOE": party.get("moe"),
            "PollSeats": party.get("projected_seats_with_threshold"),
            # Seat counts are integers; present the median as a whole seat.
            "Median": int(np.floor(np.median(values) + 0.5)) if values.size else 0,
            "90% CI range": (f"{np.percentile(values, 5):.0f}–{np.percentile(values, 95):.0f}"
                             if values.size else "0–0"),
            "Colour": party.get("color") or "#4C78A8",
            "_values": values,
        })
    return pd.DataFrame(rows)

def display_table(frame: pd.DataFrame):
    """Render a static results table with party colour and compact polling details."""
    colours = (frame["Colour"].fillna("#4C78A8").astype(str)
               if "Colour" in frame else pd.Series(dtype=str))
    shown = frame.drop(
        columns=["Party ID", "_values", "Party name", "Colour", "MOE"], errors="ignore"
    ).copy()

    # Sort each scope by its own polling support, highest first. Missing values
    # belong at the bottom (and are normally filtered from regional tables).
    if "Poll%" in shown:
        sort_values = pd.to_numeric(shown["Poll%"], errors="coerce")
        shown = shown.assign(_poll_sort=sort_values).sort_values(
            "_poll_sort", ascending=False, na_position="last", kind="stable"
        ).drop(columns="_poll_sort")
        colours = colours.loc[shown.index]

    if shown.empty:
        return

    colour_by_index = colours.to_dict()
    shown.insert(0, "", ["" for _ in range(len(shown))])
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in shown.columns)
    rows = []
    for index, row in shown.iterrows():
        cells = []
        for column, value in row.items():
            if column == "":
                colour = colour_by_index.get(index, "#4C78A8")
                if not re.fullmatch(r"#[0-9A-Fa-f]{6}", colour):
                    colour = "#4C78A8"
                content = f'<span class="party-colour-swatch" style="background:{colour}"></span>'
                cells.append(f'<td class="party-colour">{content}</td>')
            elif column == "Poll%" and not pd.isna(value):
                content = f'<span class="poll-value">{float(value):.1f}%</span>'
                moe = frame.at[index, "MOE"] if "MOE" in frame else None
                if moe is not None and not pd.isna(moe):
                    content += f'<span class="poll-moe">±{float(moe):.1f}%</span>'
                cells.append(f"<td>{content}</td>")
            else:
                text = "N/A" if pd.isna(value) else str(value)
                cells.append(f"<td>{escape(text)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    table_html = f"""
<style>
.poll-results {{ width: 100%; border-collapse: collapse; }}
.poll-results th, .poll-results td {{
    padding: 0.45rem 0.7rem;
    border-bottom: 1px solid rgba(128, 128, 128, 0.25);
    text-align: left;
}}
.poll-results .party-colour {{
    width: 3rem; min-width: 3rem; max-width: 3rem;
    padding-left: 0; padding-right: 0;
}}
.party-colour-swatch {{ display: block; width: 3rem; height: 1.75rem; }}
.poll-moe {{ display: block; color: #808080; font-size: 0.65em; line-height: 1.15; }}
</style>
<table class="poll-results">
<thead><tr>{headers}</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
"""
    st.markdown(table_html, unsafe_allow_html=True)


def chart_for_party(frame: pd.DataFrame, selected_party_id: int):
    """Plot one party's simulation distribution, identified by stable party ID."""
    matches = frame.loc[frame["Party ID"] == int(selected_party_id)]
    if matches.empty:
        st.warning("No simulation data is available for this party in this scope.")
        return
    row = matches.iloc[0]
    values = np.asarray(row["_values"], dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        st.warning("No valid simulated seat values are available for this party.")
        return
    histogram = pd.DataFrame({"Seats": values.astype(int)})
    spread = int(values.max() - values.min() + 1)
    fig = px.histogram(
        histogram, x="Seats", nbins=min(60, max(1, spread)),
        title=f'{row["Party name"]}: simulated seat distribution',
        labels={"count": "Simulations", "Seats": "Seats"},
        color_discrete_sequence=[row["Colour"] if isinstance(row["Colour"], str) else "#4C78A8"],
    )
    fig.update_layout(showlegend=False, bargap=0.04)
    st.plotly_chart(fig, width="stretch")


# --------------------------------- App -------------------------------------

st.set_page_config(page_title="PTR Poll Simulator", page_icon="📊", layout="wide")

try:
    nations = get_json(API_BASE)
except Exception as exc:
    st.error(f"Could not load the nation list: {exc}")
    st.stop()

if not isinstance(nations, list) or not nations:
    st.error("The API returned no nations.")
    st.stop()

nations_by_id = {int(nation["id"]): nation for nation in nations}
nation_ids = list(nations_by_id)

with st.sidebar:
    selected_nation_id = st.selectbox(
        "Nation",
        nation_ids,
        format_func=lambda nation_id: nations_by_id[nation_id]["name"],
    )
    selected_nation = nations_by_id[selected_nation_id]
    st.header("Simulation settings")
    simulations = st.number_input("Number of simulations", min_value=100, max_value=100000,
                                  value=10000, step=1000)
    st.caption("The confidence interval shown in the results is the central 90% simulation interval.")

nation_base = f"{API_BASE}/{selected_nation_id}"
st.title(f"{selected_nation['name']} National Poll Monte Carlo Simulator")
st.caption("Poll data, constitutional seat allocation and electoral-list rules from the PTR API.")

try:
    poll_index = get_json(f"{nation_base}/polls/national?offset=0&limit=40")
except Exception as exc:
    st.error(f"Could not load the poll index: {exc}")
    st.stop()

poll_items = poll_index.get("items", []) if isinstance(poll_index, dict) else []
if not poll_items:
    st.warning("The API returned no national polls.")
    st.stop()

polls_by_id = {int(poll["id"]): poll for poll in poll_items}
poll_id = st.selectbox(
    "Choose a poll",
    list(polls_by_id),
    format_func=lambda poll_id: (
        f'{polls_by_id[poll_id].get("game_month", "unknown month")} · '
        f'{polls_by_id[poll_id].get("created_at", "")[:10]} · Poll {poll_id}'
    ),
    key=f"poll_{selected_nation_id}",
)
with st.expander("Use a poll ID instead"):
    use_manual_poll_id = st.checkbox(
        "Override the selected poll",
        key=f"use_manual_poll_{selected_nation_id}",
    )
    if use_manual_poll_id:
        poll_id = st.number_input(
            "Poll ID",
            min_value=1,
            value=int(poll_id),
            step=1,
            key=f"manual_poll_{selected_nation_id}",
        )

col_a, col_b = st.columns([1, 1])
with col_a:
    selected_poll = polls_by_id.get(poll_id)
    if selected_poll:
        poll_description = (
            f'{selected_poll.get("game_month", "unknown month")} · Poll {poll_id}'
        )
    else:
        poll_description = f"Poll {poll_id} (manual)"
    st.write(f"Selected poll: **{poll_description}**")
with col_b:
    run = st.button("Run Monte Carlo simulation", type="primary", width="stretch")

if run:
    try:
        with st.spinner("Fetching poll, constitution and electoral lists..."):
            poll = get_json(f"{nation_base}/polls/{poll_id}")
            constitution = get_json(f"{nation_base}/constitution")
            joint_list_data = get_json(f"{nation_base}/electoral-lists")
        try:
            government_data = get_json(f"{nation_base}/government")
        except Exception as exc:
            government_data = None
            st.warning(f"Could not load government data; party results will still be available. Details: {exc}")
    except Exception as exc:
        st.error(f"Could not retrieve required poll, constitution or electoral-list data: {exc}")
        st.stop()

    poll_month = poll.get("game_month")
    if not poll_month:
        st.error("The poll response has no game_month. Cannot determine which joint lists existed.")
        st.stop()

    # The API supplies the current constitution. Warn when it post-dates the poll,
    # because the app deliberately does not accept historical local overrides.
    constitution_month = constitution.get("adopted_at")
    if month_key(constitution_month) and month_key(poll_month) and month_key(constitution_month) > month_key(poll_month):
        st.warning(
            f"The fetched constitution was adopted in {constitution_month}, after this poll "
            f"({poll_month}). The simulator will use the current constitution returned by the API, "
            "so historical seat apportionment and thresholds may differ."
        )

    provisions = provisions_map(constitution)
    threshold = get_provision_value(provisions, "electoral_threshold_pct", poll.get("threshold_pct", 5))
    if threshold is None:
        threshold = float(poll.get("threshold_pct", 5) or 5)
    mode = derive_electoral_mode(constitution, poll)
    allocation_algorithm = derive_allocation_algorithm(constitution)
    list_threshold = get_provision_value(
        provisions, "coalition_threshold_pct", poll.get("coalition_threshold_pct")
    )
    if list_threshold is not None:
        list_threshold = float(list_threshold)

    government = latest_government(government_data)
    government_party_ids = current_government_party_ids(government)

    joint_lists = active_joint_lists(joint_list_data, poll_month)
    if not joint_lists:
        st.info("No qualifying joint lists were found as active at this poll date. Parties will compete individually.")

    # Use the list identities recorded in the poll, but only pool them if the
    # corresponding list was active at the poll date.
    known_list_ids = {
        int(x) for p in poll.get("parties", [])
        for x in (p.get("electoral_list_id"), p.get("partial_list_id")) if x is not None
    }
    joint_lists = {lid: data for lid, data in joint_lists.items() if lid in known_list_ids}

    st.subheader("Poll and electoral rules")
    st.write(
        f"**Poll:** {poll_id} · **Game month:** {poll_month} · "
        f"**Seats:** {poll.get('total_seats', 'unknown')} · "
        f"**Constitutional allocation:** {mode} · "
        f"**Seat algorithm:** {allocation_algorithm}"
    )
    st.write(f"**Threshold:** {threshold:g}% · **Joint-list threshold:** "
             f"{'same as national threshold' if list_threshold is None else f'{list_threshold:g}%'}")
    st.write(f"**Active joint lists used:** {', '.join(x.get('name', str(k)) for k, x in joint_lists.items()) or 'None'}")

    # Fresh entropy-backed generator for every run. No seed is exposed to players.
    rng = np.random.default_rng()
    national_runs = []
    region_runs: dict[str, list[dict[int, int]]] = {}
    progress = st.progress(0, text="Simulating seat allocations...")
    for i in range(int(simulations)):
        national, regions = simulate_once(
            poll, constitution, joint_lists, mode, float(threshold), list_threshold,
            allocation_algorithm, rng
        )
        national_runs.append(national)
        for region_name, result in regions.items():
            region_runs.setdefault(region_name, []).append(result)
        if (i + 1) % max(1, int(simulations) // 100) == 0:
            progress.progress((i + 1) / int(simulations), text=f"Simulation {i + 1:,} / {int(simulations):,}")
    progress.empty()

    st.session_state["simulation_output"] = {
        "poll": poll,
        "national_runs": national_runs,
        "region_runs": region_runs,
        "poll_id": poll_id,
        "nation_id": selected_nation_id,
        "mode": mode,
        "allocation_algorithm": allocation_algorithm,
        "simulations": int(simulations),
        "government": government,
        "government_party_ids": sorted(government_party_ids),
    }

saved = st.session_state.get("simulation_output")
if saved and saved.get("nation_id") == selected_nation_id:
    poll = saved["poll"]
    national_runs = saved["national_runs"]
    region_runs = saved["region_runs"]
    poll_id = saved["poll_id"]
    allocation_algorithm = saved.get("allocation_algorithm", "Sainte-Lague")
    mode = saved.get("mode", "District threshold")
    parties = poll.get("parties", [])
    national_party_meta = {int(p["party_id"]): p for p in parties}
    district_parties_by_name = {
        district.get("name", f"District {district.get('district_id', '')}"):
            district.get("parties", [])
        for district in poll.get("districts", [])
    }
    st.success(f"Showing {saved['simulations']:,} saved simulations for poll {poll_id}.")

    government = saved.get("government")
    government_party_ids = set(saved.get("government_party_ids", []))
    if government and government_party_ids:
        st.subheader("Government vs opposition")
        government_name = government.get("display_name") or f"Government formed {government.get('formed_at', 'unknown')}"
        st.caption(
            f"Based on the most recently formed government: **{government_name}** "
            f"({government.get('formed_at', 'formation date unknown')}). "
            "Parties are included when they have at least one currently serving government member."
        )
        gov_summary = build_government_opposition_summary(
            national_runs, government_party_ids, int(poll.get("total_seats", 0) or 0)
        )
        if not gov_summary.empty:
            st.dataframe(gov_summary, hide_index=True, width="stretch")
        else:
            st.info("Government seat totals could not be calculated.")
    else:
        st.warning("The government endpoint did not provide a usable current government or serving party membership.")

    tabs = st.tabs(["National results"] + list(region_runs.keys()))
    scopes = [("National results", national_runs)] + list(region_runs.items())
    for tab, (scope_name, runs) in zip(tabs, scopes):
        with tab:
            st.subheader(scope_name)
            # Regional tabs must display regional polling inputs, not repeat the
            # national poll figures. The district party rows contain local support,
            # MOE and projected-seat estimates.
            if scope_name == "National results":
                scope_parties = parties
            else:
                # Regional rows carry local polling figures, but some API
                # responses omit the party name/colour there. Fill only those
                # descriptive fields from the national party record; never
                # replace regional support, MOE, or projected seats.
                scope_parties = []
                for local_party in district_parties_by_name.get(scope_name, []):
                    # A null local support estimate denotes a party not contesting
                    # this region in the poll data. Do not show it in this table.
                    if local_party.get("support_pct") is None:
                        continue
                    pid = int(local_party["party_id"])
                    national_party = national_party_meta.get(pid, {})
                    merged = {**national_party, **local_party}
                    for field in ("party_name", "name", "abbreviation", "color"):
                        if not merged.get(field) and national_party.get(field):
                            merged[field] = national_party[field]
                    scope_parties.append(merged)
            summary = build_summary(runs, scope_parties, scope_name)
            summary = add_largest_party_probability(summary, runs)
            display_table(summary)
            if not summary.empty:
                options = {
                    int(row["Party ID"]): f'{row["Party"]} · {int(row["Party ID"])}'
                    for _, row in summary.iterrows()
                }
                selected_party_id = st.selectbox(
                    f"View party distribution · {scope_name}",
                    list(options),
                    format_func=lambda pid: options[pid],
                    key=f"chart_{selected_nation_id}_{scope_name}"
                )
                chart_for_party(summary, selected_party_id)
            st.download_button(
                f"Download {scope_name} results (CSV)",
                summary.drop(columns=["Party ID", "_values", "Colour"], errors="ignore").to_csv(index=False),
                file_name=f"poll_{poll_id}_{scope_name.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                key=f"download_{selected_nation_id}_{scope_name}"
            )

    with st.expander("Simulation assumptions and limitations"):
        st.markdown(f"""
- Each party's poll estimate is sampled from a normal distribution with standard deviation `MOE / 1.96`.
- Negative draws are rejected, then all sampled party shares in the relevant pool are normalised to 100%.
- District simulations use district-level poll estimates and MOEs.
- District seat counts come from the constitution's `apportionment` array.
- Thresholds are applied to standalone parties or pooled qualifying joint lists.
- Constitutional allocation mode: **{mode}**.
- Seat allocation algorithm: **{allocation_algorithm}**.
- Sainte-Laguë uses odd divisors (1, 3, 5, ...); D'Hondt uses divisors (1, 2, 3, ...); Hare-largest remainder uses the Hare quota and largest remainders.
- Joint-list seats are split by each member party's `ratio_pct` in the active joint-list data. Ratios are renormalised over only the member parties running in the relevant district, then whole seats are assigned by the largest-remainder method.
- The API's poll-level `projected_seats_with_threshold` is shown as `PollSeats`; it is not used to force the simulation toward the API projection.
- Chance largest party counts simulations in which a party outright holds the most seats.
- Chance tied for largest counts simulations in which a party shares the largest seat count with one or more other parties.
- Government/opposition totals use the parties with currently serving members in the most recently formed government returned by the API. Majority means at least half the chamber plus one seat.
- The constitution and electoral-list data are fetched from the API. For older polls, the current constitution may differ from the rules that applied at the poll date.
- A fresh, hidden random seed is generated for every simulation run, so rerunning the same settings can produce different results.
        """)
else:
    st.info("Choose a poll and simulation settings, then click **Run Monte Carlo simulation**.")
