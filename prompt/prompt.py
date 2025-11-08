from textwrap import dedent

def render_start_state_to_text(start_state):
    return str(start_state)

def build_prompt(start_state):
    state = render_start_state_to_text(start_state)
    return dedent(f"""\
    You are a math expert and your goal is to prove a theorem using Lean4. Your output should be a single tactic in JSON, formatted like this:

        {{
            "tactic": "{{YOUR_TACTIC_HERE}}"
        }}

    Do not provide commentary or explanations, just one string tactic in JSON format.

    Here is the problem you need to solve:
    {state}
    """).strip()
