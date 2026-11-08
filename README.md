# makerspace-analytics MCP server (HW5 Q4)

Five tools in two groups. HVAC (HW4): `discover_thermal_model`, `discover_cop_law`, `daily_heating_energy`. 3D printing (HW5): `part_load_check`, `failure_cost_ledger`.
`hw4_calcs.py` is the HW4 notebook math, unchanged.

## Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate   # Python >= 3.10
pip install -r requirements.txt
```
## Run / test
```bash
python demo_client.py                      # MCP client over stdio; writes demo_transcript.txt
npx @modelcontextprotocol/inspector .venv/bin/python server.py   # MCP Inspector UI
```
Claude Code / Desktop: `claude mcp add makerspace -- /abs/path/.venv/bin/python /abs/path/server.py`

Invalid input returns a normal MCP tool result with `isError: true` and a plain-language message.
