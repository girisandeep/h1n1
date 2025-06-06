# langgraph_agent.py
from dotenv import load_dotenv
load_dotenv()

import logging
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt.chat_agent_executor import create_tool_calling_executor

# Turn on DEBUG logging so that we see LangChain internals if needed
# logging.basicConfig(level=logging.INFO)
# logging.getLogger("langchain").setLevel(logging.DEBUG)
# logging.getLogger("langchain_core").setLevel(logging.DEBUG)
# logging.getLogger("openai").setLevel(logging.DEBUG)

# Import your actual model functions
from h1n1_agent import train_h1n1_model, predict_h1n1_adoption

@tool("train_h1n1_model", description="Train the H1N1 model from a CSV folder. the default value should be data.", return_direct=True)
def train_tool_wrapper(csv_folder: str) -> dict:
    # Manually print the function call and arguments
    print("\n--- TOOL CALL START: train_h1n1_model ---")
    print("  args:", {"csv_folder": csv_folder})
    try:
        result = train_h1n1_model(csv_folder)
        print("--- TOOL CALL END:   train_h1n1_model →", result, "\n")
        return result
    except Exception as ex:
        print("--- TOOL CALL ERROR: train_h1n1_model raised", repr(ex), "\n")
        raise

@tool(
    "predict_h1n1_adoption",
    description=(
        "Predict whether someone will adopt the H1N1 vaccine. "
        "Each field below is a separate argument; we will reassemble them into a single dict."
    ),
    return_direct=True,
)
def predict_tool_wrapper(
    h1n1_concern: int,
    h1n1_knowledge: int,
    behavioral_face_mask: int,
    behavioral_large_gatherings: int,
    behavioral_wash_hands: int,
    behavioral_outside_home: int,
    behavioral_touch_face: int,
    child_under_6_months: int,
    health_worker: int,
    opinion_h1n1_vacc_effective: int,
    opinion_h1n1_risk: int,
    opinion_h1n1_sick_from_vacc: int,
    age_group: str,
    education: str,
    race: str,
    sex: str,
    income_poverty: str,
    marital_status: str,
    rent_or_own: str,
    employment_status: str,
    census_msa: str,
    hhs_geo_region: str,
    behavioral_social_distance: int,
    employment_industry: str,
    employment_occupation: str,
) -> dict:
    # Reassemble arguments into the dict that predict_h1n1_adoption expects
    user_input = {
        "h1n1_concern": h1n1_concern,
        "h1n1_knowledge": h1n1_knowledge,
        "behavioral_face_mask": behavioral_face_mask,
        "behavioral_large_gatherings": behavioral_large_gatherings,
        "behavioral_wash_hands": behavioral_wash_hands,
        "behavioral_outside_home": behavioral_outside_home,
        "behavioral_touch_face": behavioral_touch_face,
        "child_under_6_months": child_under_6_months,
        "health_worker": health_worker,
        "opinion_h1n1_vacc_effective": opinion_h1n1_vacc_effective,
        "opinion_h1n1_risk": opinion_h1n1_risk,
        "opinion_h1n1_sick_from_vacc": opinion_h1n1_sick_from_vacc,
        "age_group": age_group,
        "education": education,
        "race": race,
        "sex": sex,
        "income_poverty": income_poverty,
        "marital_status": marital_status,
        "rent_or_own": rent_or_own,
        "employment_status": employment_status,
        "census_msa": census_msa,
        "hhs_geo_region": hhs_geo_region,
        "behavioral_social_distance": behavioral_social_distance,
        "employment_industry": employment_industry,
        "employment_occupation": employment_occupation,
    }

    # Print the assembled dict right before calling the model
    # print("\n--- TOOL CALL START: predict_h1n1_adoption ---")
    # print("  user_input:", user_input)
    try:
        output = predict_h1n1_adoption(user_input)
        print("--- TOOL CALL END:   predict_h1n1_adoption →", output, "\n")
        return output
    except Exception as ex:
        # print("--- TOOL CALL ERROR: predict_h1n1_adoption raised", repr(ex), "\n")
        raise

def read_until_two_blank_lines():
    """
    Reads lines from the user until two blank lines are entered in a row.
    Returns the collected text (excluding the terminating blank lines) as a single string.
    """
    lines = []
    blank_count = 0

    while True:
        try:
            line = input(">> ")
        except EOFError:
            break

        if line.strip() == "":
            blank_count += 1
        else:
            blank_count = 0

        # Stop when two blank lines in a row
        if blank_count >= 2:
            break

        lines.append(line)

    return "\n".join(lines)

def main():
    # 1) Create ChatOpenAI with verbose + StdOutCallbackHandler
    llm = ChatOpenAI(
        temperature=0,
        verbose=True,                          # ← prints raw prompt + raw completion
        callbacks=[StdOutCallbackHandler()],   # ← prints each tool‐start and tool‐end at token time
    )

    # 2) Register your two tool wrappers
    tools = [train_tool_wrapper, predict_tool_wrapper]

    # 3) Create the executor WITHOUT verbose/callbacks arguments (they’re not supported)
    app = create_tool_calling_executor(
        llm,
        tools
    )

    print("H1N1‐Vaccine Agent (type 'exit' or 'quit' to stop)")
    print("Press ENTER twice to complete the input.")
    while True:
        user_text = read_until_two_blank_lines().strip()
        if user_text.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        human_msg = HumanMessage(content=user_text)
        try:
            result = app.invoke({"messages": [human_msg]})
            print(result)
            # # 4) Print every message in the final exchange
            # print("=== FULL CHAT EXCHANGE ===")
            # for i, msg in enumerate(result["messages"], start=1):
            #     role_name = msg.__class__.__name__  # e.g. HumanMessage, AIMessage, ToolResponseMessage
            #     print(f"{i:02d}) [{role_name}] {msg.content}")
            # print("=== END EXCHANGE ===\n")

        except KeyboardInterrupt:
            # If you hit Ctrl+C during input(), it will drop here.
            print("\nInterrupted by user; exiting.")
            break

        except Exception as e:
            print(f"[Error] {e}")
            break

if __name__ == "__main__":
    main()
