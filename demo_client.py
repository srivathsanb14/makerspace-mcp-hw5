"""MCP client demo: launches server.py over stdio, lists tools, and records calls to demo_transcript.txt."""
import asyncio, json, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

CALLS = [
    ("discover_thermal_model (original)", "discover_thermal_model", {"threshold": 0.1, "window_pts": 24}),
    ("discover_cop_law (original)", "discover_cop_law", {}),
    ("daily_heating_energy (original)", "daily_heating_energy", {"T_out_C": 5, "T_set_C": 21, "T_supply_C": 40, "price_per_kWh": 0.17}),
    ("part_load_check (NEW)", "part_load_check", {"material": "PLA", "load_N": 40, "span_mm": 80, "width_mm": 20, "height_mm": 10, "support": "cantilever", "orientation": "flat", "resumed_after_pause": "yes_no_reheat"}),
    ("failure_cost_ledger (NEW)", "failure_cost_ledger", {"n_printers": 20, "failure_rate": 0.138, "recoverable_fraction": 0.5}),
    ("INVALID: schema range (T_out_C=25 > max 20)", "daily_heating_energy", {"T_out_C": 25}),
    ("INVALID: unknown material (enum)", "part_load_check", {"material": "WOOD", "load_N": 40, "span_mm": 80, "width_mm": 20, "height_mm": 10}),
    ("INVALID: schema range (failure_rate=0.9 > 0.6)", "failure_cost_ledger", {"n_printers": 20, "failure_rate": 0.9}),
]

async def main():
    out = []
    params = StdioServerParameters(command=sys.executable, args=["server.py"])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            out.append("tools/list -> " + ", ".join(t.name for t in tools.tools))
            for label, name, args in CALLS:
                res = await s.call_tool(name, args)
                body = res.structuredContent if res.structuredContent else "".join(c.text for c in res.content if hasattr(c, "text"))
                out.append(f"\n### {label}\ncall_tool({name}, {json.dumps(args)})\nisError={res.isError}\n"
                           + (json.dumps(body, indent=2) if not isinstance(body, str) else body))
    text = "\n".join(out)
    open("demo_transcript.txt", "w").write(text)
    print(text)

asyncio.run(main())
