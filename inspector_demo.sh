#!/bin/sh
# Calls each required tool through the official MCP Inspector (CLI mode). Output -> inspector_transcript.txt
I="npx -y @modelcontextprotocol/inspector --cli .venv/bin/python server.py --method tools/call"
{
echo "### Inspector: original tool daily_heating_energy"; $I --tool-name daily_heating_energy --tool-arg T_out_C=5 2>/dev/null
echo "### Inspector: NEW tool part_load_check"; $I --tool-name part_load_check --tool-arg material=PETG --tool-arg load_N=30 --tool-arg span_mm=100 --tool-arg width_mm=20 --tool-arg height_mm=8 --tool-arg orientation=upright 2>/dev/null
echo "### Inspector: NEW tool failure_cost_ledger"; $I --tool-name failure_cost_ledger --tool-arg n_printers=3 --tool-arg jobs_per_printer_per_week=5 --tool-arg recoverable_fraction=0.5 2>/dev/null
echo "### Inspector: INVALID input (T_out_C=25)"; $I --tool-name daily_heating_energy --tool-arg T_out_C=25 2>/dev/null
} > inspector_transcript.txt
