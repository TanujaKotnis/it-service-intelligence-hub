"""Shared helper utilities for the Streamlit app."""

import json
from pathlib import Path
import streamlit as st

RESULTS_PATH = Path(__file__).parent.parent / "fine_tuning" / "training_results.json"
FINE_TUNED_PATH = Path(__file__).parent.parent / "fine_tuning" / "saved_model"


def load_training_results() -> dict | None:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return None


def is_fine_tuned() -> bool:
    return FINE_TUNED_PATH.exists()


def confidence_color(score: float) -> str:
    if score >= 80:
        return "green"
    elif score >= 50:
        return "orange"
    return "red"


def render_metric_card(label: str, value: str, delta: str = "", color: str = "blue"):
    """Render a simple styled metric using st.metric."""
    st.metric(label=label, value=value, delta=delta if delta else None)


def render_score_bar(label: str, score: float, max_width: int = 300):
    """Render a progress-bar-style score visualization."""
    filled = int(score / 100 * max_width)
    color = confidence_color(score)
    st.markdown(
        f"""
        <div style="margin-bottom:4px;">
          <span style="font-size:0.85rem;">{label}</span>
          <span style="float:right;font-size:0.85rem;color:{color};font-weight:600;">{score:.1f}%</span>
        </div>
        <div style="background:#e8e8e8;border-radius:6px;height:10px;margin-bottom:10px;">
          <div style="background:{color};width:{score}%;height:10px;border-radius:6px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


SAMPLE_TICKETS = [
    {
        "text": "I cannot connect to the VPN from home. The Cisco AnyConnect client gives an authentication error every time I try.",
        "label": "network_connectivity",
    },
    {
        "text": "SAP GUI is not launching. I get a runtime error when I double-click the icon. I have an urgent report due today.",
        "label": "software_installation",
    },
    {
        "text": "My laptop screen has horizontal lines and is flickering badly. Very difficult to work on anything.",
        "label": "hardware_malfunction",
    },
    {
        "text": "I received a suspicious email asking me to click a link to verify my password. I did not click it but need advice.",
        "label": "security_incident",
    },
    {
        "text": "I forgot my password and my account is locked. I have a client presentation in one hour and cannot log in.",
        "label": "account_access",
    },
    {
        "text": "Outlook is not syncing emails since this morning. Last sync was 4 hours ago and I am missing important client communications.",
        "label": "email_issues",
    },
    {
        "text": "The printer on floor 3 is offline and not printing any jobs. The print queue has built up with 20+ pending jobs.",
        "label": "printer_problems",
    },
]

LONG_TICKET_EXAMPLE = """
Dear IT Support Team,

I am writing to report an urgent and highly disruptive issue that has been affecting my productivity
and that of my entire team since early this morning around 8:30 AM CET. I am a senior project manager
in the Digital Industries division and I lead a team of 12 engineers who all rely heavily on our
corporate network infrastructure to access SAP, SharePoint, and our internal engineering tools.

The issue began when I arrived at the office and attempted to connect my Dell Latitude 7430 laptop
to the docking station. Upon connection, the laptop failed to obtain an IP address from the DHCP
server. I confirmed this by running "ipconfig" in the command prompt, which showed "169.254.x.x"
as the IPv4 address, clearly indicating a DHCP failure. I then tried connecting directly via an
Ethernet cable bypassing the docking station, but the problem persisted. At this point, I noticed
that two of my colleagues in adjacent workstations were experiencing the same issue, suggesting
this may be a switch or DHCP server issue rather than a problem specific to my device.

I have already attempted the following troubleshooting steps without success: releasing and
renewing the IP address using "ipconfig /release" and "ipconfig /renew", restarting the network
adapter, restarting my laptop twice, trying multiple Ethernet ports on the docking station, and
contacting the building maintenance team to check if there were any physical network disruptions.

Currently, myself and at least 7 other colleagues on this floor cannot access any corporate
systems. We have a critical project milestone review with our customer Volkswagen AG scheduled
at 2:00 PM CET today and require immediate resolution. Without network access, we cannot prepare
the required presentations or access the project documentation in SharePoint.

Please treat this as a Priority 1 incident and escalate to the Network Operations Center immediately.
I can be reached at +49 89 636 12345 or via Teams once my connection is restored.

Best regards,
[Employee Name]
Department: Digital Industries — Automation
Floor: Building 4, Floor 2, Munich Campus
"""
