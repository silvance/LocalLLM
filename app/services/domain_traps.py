"""Small, curated domain trap cards for code-generation prompts.

These are not RAG docs. They are short "known failure mode" cards
injected before the first writer pass so local models don't start from
an impossible architecture and then spend review rounds defending it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrapCard:
    id: str
    triggers: tuple[str, ...]
    constraints: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]

    def render(self) -> str:
        constraints = "\n".join(f"- {c}" for c in self.constraints)
        criteria = "\n".join(f"- {c}" for c in self.acceptance_criteria)
        return (
            f"Trap card: {self.id}\n"
            f"Hard constraints:\n{constraints}\n"
            f"Acceptance criteria:\n{criteria}"
        )


WIRELESS_BLE_WIFI_LINUX_SBC = TrapCard(
    id="wireless_ble_wifi_linux_sbc",
    triggers=(
        "ble",
        "bluetooth",
        "wi-fi",
        "wifi",
        "sniffer",
        "packet capture",
        "raspberry pi",
        "orange pi",
        "linux sbc",
    ),
    constraints=(
        "Do not capture BLE from wlan0, wlan0mon, mon0, wlp*, or any Wi-Fi monitor-mode interface.",
        "Do not use scapy.sniff(iface=\"hci0\") as a Pi-native BLE sniffer.",
        "Do not use pyshark.LiveCapture(interface=\"hci0\") unless the capture source is explicitly verified.",
        "Do not check BTLE_ADV inside a Wi-Fi Scapy packet handler.",
        "Use separate Wi-Fi and BLE backends.",
        "Wi-Fi backend may use monitor-mode Wi-Fi tooling such as iw, tshark, tcpdump, airodump-ng, or Scapy on a verified monitor-mode interface.",
        "BLE backend should use BlueZ tools such as bluetoothctl, btmgmt, or btmon for Pi-native discovery/HCI logging.",
        "Full over-the-air BLE sniffing requires dedicated BLE sniffer hardware such as nRF52840/nRF Sniffer, Ubertooth, or commercial tools.",
        "Do not simulate BLE packets.",
    ),
    acceptance_criteria=(
        "Code accepts separate Wi-Fi and BLE options/backends.",
        "Wi-Fi prerequisites are checked or clearly reported.",
        "BLE backend does not depend on wlan0/mon0/Scapy BLE packet layers.",
        "If external tools are started, their output is consumed or their artifact path is reported.",
    ),
)


TRAP_CARDS: tuple[TrapCard, ...] = (WIRELESS_BLE_WIFI_LINUX_SBC,)


RISKY_DOMAIN_KEYWORDS: tuple[str, ...] = (
    "hardware",
    "network",
    "packet",
    "radio",
    "forensic",
    "sniffer",
    "capture",
    "embedded",
    "raspberry pi",
    "linux",
    "driver",
    "serial",
    "bluetooth",
    "ble",
    "wifi",
    "wi-fi",
    "esp32",
)


def classify(prompt: str) -> list[TrapCard]:
    text = prompt.casefold()
    matches: list[TrapCard] = []
    for card in TRAP_CARDS:
        hits = sum(1 for trigger in card.triggers if trigger in text)
        if hits >= 2:
            matches.append(card)
    return matches


def needs_feasibility_plan(prompt: str) -> bool:
    text = prompt.casefold()
    return any(keyword in text for keyword in RISKY_DOMAIN_KEYWORDS)


def builder_context(prompt: str) -> str:
    """Return extra instructions for the writer, or ``""`` if none apply."""
    cards = classify(prompt)
    if not cards and not needs_feasibility_plan(prompt):
        return ""

    parts: list[str] = [
        "Before generating code for this external-interface task, perform an internal feasibility check.",
        "Do not assume similar technologies share an interface, driver, protocol layer, packet format, command syntax, or library support.",
        "Identify the target OS/hardware, required tools/services, privileges, exact API/interface, setup/teardown, version assumptions, and anything unverifiable from the prompt.",
        "If real implementation requires external tools or dedicated hardware, wrap/report that honestly. Do not simulate core requested functionality unless the user explicitly asked for a mock/demo.",
        "Return exactly one complete fenced ```python``` code block.",
    ]
    parts.extend(card.render() for card in cards)
    return "\n\n".join(parts)


def wrap_builder_prompt(prompt: str) -> str:
    context = builder_context(prompt)
    if not context:
        return prompt
    return f"{context}\n\nOriginal user request:\n{prompt}"

