from app.services.domain_traps import builder_context, classify, wrap_builder_prompt


def test_ble_wifi_trap_classifies_linux_sbc_sniffer() -> None:
    cards = classify("Build a quick BLE/Wi-Fi sniffer using a Raspberry Pi 4 Model B.")
    assert [c.id for c in cards] == ["wireless_ble_wifi_linux_sbc"]


def test_builder_context_includes_ble_wifi_constraints() -> None:
    text = builder_context("Build a BLE and Wi-Fi sniffer for Raspberry Pi.")
    lowered = text.lower()
    assert "trap card: wireless_ble_wifi_linux_sbc" in lowered
    assert "do not capture ble from wlan0" in lowered
    assert "use separate wi-fi and ble backends" in lowered
    assert "do not simulate ble packets" in lowered


def test_wrap_builder_prompt_is_noop_for_simple_tasks() -> None:
    prompt = "Write a function that adds two numbers."
    assert wrap_builder_prompt(prompt) == prompt

