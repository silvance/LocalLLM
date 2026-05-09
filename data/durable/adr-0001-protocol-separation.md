---
id: adr-0001
title: Wi-Fi and BLE belong on separate backends
status: durable
valid_from: 2026-05-09
valid_to:
transaction_time: 2026-05-09
supersedes: []
superseded_by:
applies_to:
  - protocol_separation
  - wifi
  - ble
  - sniffer_tasks
provenance:
  - PR-38 review-loop hardening (May 2026)
  - reviewer-blocker repeat-pattern in BLE/Wi-Fi sniffer task
---

If the task involves BOTH Wi-Fi and BLE: implement each protocol in a
SEPARATE function / class / module. Do not share an interface, capture
loop, or `sniff()` call across protocols.

BLE capture MUST use BlueZ tooling: `btmon`, `bluetoothctl --monitor`,
the mgmt-API, or a `socket.AF_BLUETOOTH` HCI raw socket. Do NOT route
BLE through `scapy.sniff()` or `pyshark.LiveCapture(interface="hci0")`
— neither path works on stock Linux without out-of-tree setup.

Wi-Fi capture goes on a monitor-mode 802.11 interface (`wlan0mon` /
`mon0`) via `scapy.sniff(iface=...)`. NEVER pass a wlan/mon interface
to a function that decodes BLE layers.

Real third-party deps (scapy, pyshark, bleak, bluetooth/pybluez) are
FINE — assume they are installed. Do not invent module names.
