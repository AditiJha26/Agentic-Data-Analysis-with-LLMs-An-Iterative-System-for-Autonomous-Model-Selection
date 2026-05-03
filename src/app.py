import streamlit as st
import pandas as pd
import json
from tools.data_tool import load_dataset, summarize_dataset
from tools.model_tool import train_models
from tools.eval_tool import summarize_evaluation
from agent.planner import generate_plan
from agent.decision import make_decision
from agent.refiner import refine_plan

st.set_page_config(page_title="Agentic Data Analyst", layout="wide")

st.title("Agentic Data Analyst")
st.markdown("An autonomous system that plans, executes, and refines ML pipelines.")

# Sidebar for Configuration
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload your CSV", type="csv")
    max_iterations = st.slider("Max Iterations", 1, 10, 5)
    run_button = st.button("Start Analysis", type="primary")

if uploaded_file and run_button:
    df = pd.read_csv(uploaded_file)
    
    # --- STEP 1: Summarize ---
    st.subheader("Dataset Overview")
    summary = summarize_dataset(df)
    col1, col2 = st.columns(2)
    col1.metric("Rows", summary["basic_info"]["num_rows"])
    col2.metric("Columns", summary["basic_info"]["num_columns"])
    st.dataframe(df.head(5))

    # --- STEP 2: Initial Plan ---
    with st.status("Agent is thinking...", expanded=True) as status:
        st.write("Generating initial plan...")
        plan = generate_plan(summary)
        st.json(plan)
        
        target_column = plan["target_column"]
        iteration = 1
        
        # Container for loop results
        results_container = st.container()

        while iteration <= max_iterations:
            st.write(f"### Iteration {iteration}")
            
            # Step 3 & 4: Train and Evaluate
            st.write("Training models and evaluating...")
            preprocessing_steps = plan.get("preprocessing", [])
            results = train_models(df, target_column, preprocessing_steps)
            eval_summary = summarize_evaluation(results)
            
            # Display metrics in the UI
            with results_container:
                st.info(f"**Iteration {iteration} Results**")
                cols = st.columns(len(results))
                for i, (model_name, metrics) in enumerate(results.items()):
                    cols[i].metric(f"{model_name} (Acc)", f"{metrics['accuracy']:.4f}")
            
            # Step 5: Decision
            decision = make_decision(eval_summary, iteration)
            st.write(f"**Decision:** {decision['decision']}")
            st.caption(f"Reason: {decision['reason']}")

            if decision["decision"] == "STOP" and iteration > 1:
                st.success("✅ Agent reached a stopping point.")
                break
            
            if iteration < max_iterations:
                st.write("🔧 Refining plan for next round...")
                refined = refine_plan(summary, plan, eval_summary)
                plan["preprocessing"] = refined.get("preprocessing", plan["preprocessing"])
                st.code(f"Updated Preprocessing: {plan['preprocessing']}")
            
            iteration += 1
        
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    st.divider()
    st.balloons()
    st.subheader("Final Performance Report")
    st.json(results)

else:
    st.info("Please upload a CSV file and click 'Start Analysis' to begin.")
