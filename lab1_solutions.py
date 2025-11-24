"""
Lab 1 Solutions: Loan Approval - Exercise Solutions
====================================================
This file contains complete solutions for all Lab 1 exercises.

Exercises Covered:
1. Change interrupt to interrupt_after
2. Add multiple review stages with compliance check
3. Implement rejection handling
4. Test different risk scenarios
5. Add state inspection and history
"""

import os
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from getpass import getpass
from dotenv import load_dotenv


# ============================================================================
# State Definition
# ============================================================================
class LoanApplicationState(TypedDict):
    applicant_name: str
    loan_amount: float
    annual_income: float
    credit_score: int
    employment_years: int
    risk_assessment: str
    compliance_status: str
    loan_decision: str
    officer_notes: str
    rejection_reason: str
    messages: Annotated[list, "conversation history"]


# ============================================================================
# LLM Setup
# ============================================================================
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found")
    return ChatGroq(model="llama-3.1-8b-instant")


# ============================================================================
# SOLUTION 1: Change to interrupt_after
# ============================================================================
def analyze_loan_application(state: LoanApplicationState) -> LoanApplicationState:
    """AI analyzes the loan application"""
    llm = get_llm()
    
    system_message = """You are a financial risk analyst. 
    Analyze loan applications and provide comprehensive risk assessment.
    
    Risk Categories:
    - LOW RISK: Credit score > 750, 5+ years employment, loan < 30% income
    - MEDIUM RISK: Credit score 650-750, 2-5 years employment, loan 30-50% income  
    - HIGH RISK: Credit score < 650, < 2 years employment, loan > 50% income"""
    
    prompt = f"""
    Loan Application Details:
    - Applicant: {state['applicant_name']}
    - Loan Amount: ${state['loan_amount']:,.2f}
    - Annual Income: ${state['annual_income']:,.2f}
    - Credit Score: {state['credit_score']}
    - Years of Employment: {state['employment_years']}
    
    Debt-to-Income Ratio: {(state['loan_amount'] / state['annual_income']):.2%}
    
    Provide detailed risk assessment with recommendation.
    """
    
    messages = [SystemMessage(content=system_message), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    print("\n" + "="*80)
    print("🤖 AI RISK ASSESSMENT")
    print("="*80)
    print(response.content)
    print("="*80 + "\n")
    
    return {
        "risk_assessment": response.content,
        "messages": state.get("messages", []) + [
            HumanMessage(content=prompt),
            AIMessage(content=response.content)
        ]
    }


def make_loan_decision(state: LoanApplicationState) -> LoanApplicationState:
    """Make final loan decision"""
    llm = get_llm()
    
    prompt = f"""
    Final Decision for: {state['applicant_name']}
    Loan Amount: ${state['loan_amount']:,.2f}
    
    Risk Assessment:
    {state['risk_assessment']}
    
    Officer Notes:
    {state.get('officer_notes', 'No additional notes')}
    
    Compliance Status:
    {state.get('compliance_status', 'Not reviewed')}
    
    Make final decision: APPROVED or DENIED with clear terms and reasoning.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    print("\n" + "="*80)
    print("✅ FINAL LOAN DECISION")
    print("="*80)
    print(response.content)
    print("="*80 + "\n")
    
    return {"loan_decision": response.content}


def solution_1_interrupt_after():
    """
    SOLUTION 1: Using interrupt_after instead of interrupt_before
    
    Key Difference:
    - interrupt_before: Pauses BEFORE the node executes
    - interrupt_after: Pauses AFTER the node executes, allowing review of results
    """
    print("\n" + "="*80)
    print("SOLUTION 1: interrupt_after Pattern")
    print("="*80 + "\n")
    
    workflow = StateGraph(LoanApplicationState)
    
    workflow.add_node("analyze_application", analyze_loan_application)
    workflow.add_node("make_decision", make_loan_decision)
    
    workflow.add_edge(START, "analyze_application")
    workflow.add_edge("analyze_application", "make_decision")
    workflow.add_edge("make_decision", END)
    
    # KEY CHANGE: interrupt_after instead of interrupt_before
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_after=["analyze_application"]  # Pause AFTER analysis completes
    )
    
    thread_config = {"configurable": {"thread_id": "loan_sol1_001"}}
    
    initial_state = {
        "applicant_name": "Maria Garcia",
        "loan_amount": 180000.0,
        "annual_income": 95000.0,
        "credit_score": 740,
        "employment_years": 6,
        "risk_assessment": "",
        "compliance_status": "",
        "loan_decision": "",
        "officer_notes": "",
        "rejection_reason": "",
        "messages": []
    }
    
    print("📋 Application Submitted\n")
    
    # Run until interrupt (after analysis)
    for event in app.stream(initial_state, thread_config, stream_mode="values"):
        pass
    
    current_state = app.get_state(thread_config)
    print("\n⏸️  INTERRUPTED AFTER ANALYSIS")
    print(f"Analysis Complete: {'Yes' if current_state.values.get('risk_assessment') else 'No'}")
    print(f"Next Node: {current_state.next}")
    print("\nOfficer can now review the completed risk assessment before proceeding.\n")
    
    # Officer reviews and adds notes
    officer_notes = "Analysis looks good. Approve with standard terms."
    app.update_state(thread_config, {"officer_notes": officer_notes})
    
    print("👤 Officer approved, resuming...\n")
    
    # Resume to make decision
    for event in app.stream(None, thread_config, stream_mode="values"):
        pass
    
    print("✅ Completed with interrupt_after pattern\n")


# ============================================================================
# SOLUTION 2: Add Multiple Review Stages (Compliance Check)
# ============================================================================
def check_compliance(state: LoanApplicationState) -> LoanApplicationState:
    """Compliance check node"""
    llm = get_llm()
    
    prompt = f"""
    Compliance Review for: {state['applicant_name']}
    Loan Amount: ${state['loan_amount']:,.2f}
    
    Check for:
    1. Regulatory compliance (loan limits, borrower eligibility)
    2. Anti-money laundering (AML) requirements
    3. Fair lending practices
    4. Documentation completeness
    
    Credit Score: {state['credit_score']}
    Employment: {state['employment_years']} years
    
    Provide compliance status: PASS or NEEDS_REVIEW with reasoning.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    print("\n" + "="*80)
    print("📋 COMPLIANCE CHECK")
    print("="*80)
    print(response.content)
    print("="*80 + "\n")
    
    return {"compliance_status": response.content}


def solution_2_multiple_interrupts():
    """
    SOLUTION 2: Multiple review stages with compliance check
    
    Flow: analyze → compliance → decision
    Interrupts: Before compliance AND before decision
    """
    print("\n" + "="*80)
    print("SOLUTION 2: Multiple Interrupt Points")
    print("="*80 + "\n")
    
    workflow = StateGraph(LoanApplicationState)
    
    # Add all nodes including compliance
    workflow.add_node("analyze_application", analyze_loan_application)
    workflow.add_node("compliance_check", check_compliance)
    workflow.add_node("make_decision", make_loan_decision)
    
    # Define flow
    workflow.add_edge(START, "analyze_application")
    workflow.add_edge("analyze_application", "compliance_check")
    workflow.add_edge("compliance_check", "make_decision")
    workflow.add_edge("make_decision", END)
    
    # Multiple interrupt points
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["compliance_check", "make_decision"]  # Two interrupt points
    )
    
    thread_config = {"configurable": {"thread_id": "loan_sol2_001"}}
    
    initial_state = {
        "applicant_name": "David Chen",
        "loan_amount": 450000.0,
        "annual_income": 120000.0,
        "credit_score": 780,
        "employment_years": 8,
        "risk_assessment": "",
        "compliance_status": "",
        "loan_decision": "",
        "officer_notes": "",
        "rejection_reason": "",
        "messages": []
    }
    
    print("📋 Application Submitted (Large Loan - $450K)\n")
    
    # Run to first interrupt (before compliance)
    for event in app.stream(initial_state, thread_config, stream_mode="values"):
        pass
    
    state_1 = app.get_state(thread_config)
    print("\n⏸️  INTERRUPT 1: Before Compliance Check")
    print(f"Next Node: {state_1.next}\n")
    
    # Risk officer approves to continue
    print("👤 Risk Officer: Approved to proceed to compliance\n")
    
    # Continue to next interrupt (before decision)
    for event in app.stream(None, thread_config, stream_mode="values"):
        pass
    
    state_2 = app.get_state(thread_config)
    print("\n⏸️  INTERRUPT 2: Before Final Decision")
    print(f"Next Node: {state_2.next}")
    print(f"Compliance Status: {state_2.values.get('compliance_status', 'N/A')}\n")
    
    # Senior officer final approval
    app.update_state(thread_config, {
        "officer_notes": "All checks passed. Approve with premium terms."
    })
    print("👤 Senior Officer: Final approval granted\n")
    
    # Complete workflow
    for event in app.stream(None, thread_config, stream_mode="values"):
        pass
    
    print("✅ Completed with multiple interrupt pattern\n")


# ============================================================================
# SOLUTION 3: Implement Rejection Handling
# ============================================================================
def review_and_decide(state: LoanApplicationState) -> Literal["approve", "reject"]:
    """Conditional edge function for approval/rejection routing"""
    # Check if officer notes contain rejection keywords
    notes = state.get('officer_notes', '').lower()
    
    if any(word in notes for word in ['reject', 'deny', 'decline']):
        return "reject"
    return "approve"


def approve_loan(state: LoanApplicationState) -> LoanApplicationState:
    """Process approved loan"""
    print("\n" + "="*80)
    print("✅ LOAN APPROVED")
    print("="*80)
    print(f"Applicant: {state['applicant_name']}")
    print(f"Amount: ${state['loan_amount']:,.2f}")
    print("="*80 + "\n")
    
    return {"loan_decision": "APPROVED", "rejection_reason": ""}


def reject_loan(state: LoanApplicationState) -> LoanApplicationState:
    """Process rejected loan"""
    llm = get_llm()
    
    prompt = f"""
    Generate a professional loan rejection letter for:
    
    Applicant: {state['applicant_name']}
    Loan Amount: ${state['loan_amount']:,.2f}
    Credit Score: {state['credit_score']}
    
    Risk Assessment Summary:
    {state['risk_assessment']}
    
    Officer Notes:
    {state.get('officer_notes', '')}
    
    Provide clear, specific rejection reasons and suggestions for improvement.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    print("\n" + "="*80)
    print("❌ LOAN REJECTED")
    print("="*80)
    print(response.content)
    print("="*80 + "\n")
    
    return {
        "loan_decision": "REJECTED",
        "rejection_reason": response.content
    }


def solution_3_rejection_handling():
    """
    SOLUTION 3: Implement conditional rejection handling
    
    Uses conditional edges to route to approval or rejection based on officer decision
    """
    print("\n" + "="*80)
    print("SOLUTION 3: Rejection Handling with Conditional Edges")
    print("="*80 + "\n")
    
    workflow = StateGraph(LoanApplicationState)
    
    workflow.add_node("analyze_application", analyze_loan_application)
    workflow.add_node("approve_loan", approve_loan)
    workflow.add_node("reject_loan", reject_loan)
    
    workflow.add_edge(START, "analyze_application")
    
    # Conditional edge based on officer decision
    workflow.add_conditional_edges(
        "analyze_application",
        review_and_decide,
        {
            "approve": "approve_loan",
            "reject": "reject_loan"
        }
    )
    
    workflow.add_edge("approve_loan", END)
    workflow.add_edge("reject_loan", END)
    
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_after=["analyze_application"]
    )
    
    # Test Case 1: Approval
    print("=" * 80)
    print("TEST CASE 1: Approved Loan")
    print("=" * 80 + "\n")
    
    thread_1 = {"configurable": {"thread_id": "loan_sol3_approve"}}
    
    state_approve = {
        "applicant_name": "Alice Johnson",
        "loan_amount": 200000.0,
        "annual_income": 100000.0,
        "credit_score": 780,
        "employment_years": 7,
        "risk_assessment": "",
        "compliance_status": "",
        "loan_decision": "",
        "officer_notes": "",
        "rejection_reason": "",
        "messages": []
    }
    
    for event in app.stream(state_approve, thread_1, stream_mode="values"):
        pass
    
    # Officer approves
    app.update_state(thread_1, {"officer_notes": "Approved. Strong profile."})
    
    for event in app.stream(None, thread_1, stream_mode="values"):
        pass
    
    # Test Case 2: Rejection
    print("\n" + "=" * 80)
    print("TEST CASE 2: Rejected Loan")
    print("=" * 80 + "\n")
    
    thread_2 = {"configurable": {"thread_id": "loan_sol3_reject"}}
    
    state_reject = {
        "applicant_name": "Bob Wilson",
        "loan_amount": 300000.0,
        "annual_income": 60000.0,
        "credit_score": 620,
        "employment_years": 1,
        "risk_assessment": "",
        "compliance_status": "",
        "loan_decision": "",
        "officer_notes": "",
        "rejection_reason": "",
        "messages": []
    }
    
    for event in app.stream(state_reject, thread_2, stream_mode="values"):
        pass
    
    # Officer rejects
    app.update_state(thread_2, {
        "officer_notes": "Reject. High debt-to-income ratio and insufficient employment history."
    })
    
    for event in app.stream(None, thread_2, stream_mode="values"):
        pass
    
    print("✅ Both approval and rejection paths tested\n")


# ============================================================================
# SOLUTION 4: Test Different Risk Scenarios
# ============================================================================
def solution_4_risk_scenarios():
    """
    SOLUTION 4: Test various risk scenarios
    
    Tests:
    - High-risk applicant (low credit, high debt ratio)
    - Medium-risk applicant
    - Low-risk applicant
    """
    print("\n" + "="*80)
    print("SOLUTION 4: Testing Different Risk Scenarios")
    print("="*80 + "\n")
    
    workflow = StateGraph(LoanApplicationState)
    workflow.add_node("analyze_application", analyze_loan_application)
    workflow.add_node("make_decision", make_loan_decision)
    
    workflow.add_edge(START, "analyze_application")
    workflow.add_edge("analyze_application", "make_decision")
    workflow.add_edge("make_decision", END)
    
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_after=["analyze_application"]
    )
    
    # Define test scenarios
    scenarios = [
        {
            "name": "HIGH RISK",
            "applicant_name": "High Risk Applicant",
            "loan_amount": 400000.0,
            "annual_income": 50000.0,
            "credit_score": 580,
            "employment_years": 1,
            "expected_risk": "HIGH RISK"
        },
        {
            "name": "MEDIUM RISK",
            "applicant_name": "Medium Risk Applicant",
            "loan_amount": 250000.0,
            "annual_income": 75000.0,
            "credit_score": 680,
            "employment_years": 3,
            "expected_risk": "MEDIUM RISK"
        },
        {
            "name": "LOW RISK",
            "applicant_name": "Low Risk Applicant",
            "loan_amount": 200000.0,
            "annual_income": 150000.0,
            "credit_score": 820,
            "employment_years": 10,
            "expected_risk": "LOW RISK"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print("\n" + "=" * 80)
        print(f"SCENARIO {i}: {scenario['name']}")
        print("=" * 80)
        print(f"Credit Score: {scenario['credit_score']}")
        print(f"Loan/Income Ratio: {(scenario['loan_amount']/scenario['annual_income']):.1%}")
        print(f"Employment: {scenario['employment_years']} years\n")
        
        thread = {"configurable": {"thread_id": f"risk_scenario_{i}"}}
        
        state = {
            "applicant_name": scenario['applicant_name'],
            "loan_amount": scenario['loan_amount'],
            "annual_income": scenario['annual_income'],
            "credit_score": scenario['credit_score'],
            "employment_years": scenario['employment_years'],
            "risk_assessment": "",
            "compliance_status": "",
            "loan_decision": "",
            "officer_notes": "",
            "rejection_reason": "",
            "messages": []
        }
        
        # Run analysis
        for event in app.stream(state, thread, stream_mode="values"):
            pass
        
        # Check risk assessment
        current = app.get_state(thread)
        assessment = current.values.get('risk_assessment', '')
        
        # Verify expected risk level appears in assessment
        if scenario['expected_risk'].lower() in assessment.lower():
            print(f"✅ Risk correctly identified as {scenario['expected_risk']}")
        else:
            print(f"⚠️  Risk assessment may differ from expected")
        
        # Complete with appropriate decision
        if scenario['credit_score'] < 650:
            app.update_state(thread, {"officer_notes": "Deny due to high risk."})
        else:
            app.update_state(thread, {"officer_notes": "Approve with conditions."})
        
        for event in app.stream(None, thread, stream_mode="values"):
            pass
    
    print("\n✅ All risk scenarios tested\n")


# ============================================================================
# SOLUTION 5: State Inspection and History
# ============================================================================
def solution_5_state_inspection():
    """
    SOLUTION 5: Advanced state inspection and history tracking
    
    Demonstrates:
    - get_state() to inspect current state
    - get_state_history() to view all checkpoints
    - Examining messages and intermediate values
    """
    print("\n" + "="*80)
    print("SOLUTION 5: State Inspection and History")
    print("="*80 + "\n")
    
    workflow = StateGraph(LoanApplicationState)
    workflow.add_node("analyze_application", analyze_loan_application)
    workflow.add_node("make_decision", make_loan_decision)
    
    workflow.add_edge(START, "analyze_application")
    workflow.add_edge("analyze_application", "make_decision")
    workflow.add_edge("make_decision", END)
    
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_after=["analyze_application"]
    )
    
    thread_config = {"configurable": {"thread_id": "loan_sol5_inspect"}}
    
    initial_state = {
        "applicant_name": "Emma Davis",
        "loan_amount": 275000.0,
        "annual_income": 92000.0,
        "credit_score": 715,
        "employment_years": 5,
        "risk_assessment": "",
        "compliance_status": "",
        "loan_decision": "",
        "officer_notes": "",
        "rejection_reason": "",
        "messages": []
    }
    
    # Run to interrupt
    for event in app.stream(initial_state, thread_config, stream_mode="values"):
        pass
    
    # INSPECT CURRENT STATE
    print("\n" + "=" * 80)
    print("📊 STATE INSPECTION AT INTERRUPT")
    print("=" * 80)
    
    current_state = app.get_state(thread_config)
    
    print(f"\n1. Basic State Info:")
    print(f"   - Checkpoint ID: {current_state.checkpoint_id}")
    print(f"   - Next Nodes: {current_state.next}")
    print(f"   - Tasks: {len(current_state.tasks)} pending")
    
    print(f"\n2. Application Details:")
    values = current_state.values
    print(f"   - Applicant: {values['applicant_name']}")
    print(f"   - Loan Amount: ${values['loan_amount']:,.2f}")
    print(f"   - Credit Score: {values['credit_score']}")
    
    print(f"\n3. Risk Assessment:")
    assessment = values.get('risk_assessment', '')
    if assessment:
        # Show first 200 characters
        print(f"   {assessment[:200]}...")
    
    print(f"\n4. Message History:")
    messages = values.get('messages', [])
    print(f"   - Total messages: {len(messages)}")
    for i, msg in enumerate(messages, 1):
        print(f"   - Message {i}: {msg.type} ({len(msg.content)} chars)")
    
    # Add officer notes and continue
    app.update_state(thread_config, {
        "officer_notes": "Reviewed and approved with standard terms."
    })
    
    for event in app.stream(None, thread_config, stream_mode="values"):
        pass
    
    # VIEW STATE HISTORY
    print("\n" + "=" * 80)
    print("📚 STATE HISTORY (All Checkpoints)")
    print("=" * 80 + "\n")
    
    history = list(app.get_state_history(thread_config))
    print(f"Total checkpoints: {len(history)}\n")
    
    for i, checkpoint in enumerate(history, 1):
        print(f"Checkpoint {i}:")
        print(f"  - ID: {checkpoint.checkpoint_id}")
        print(f"  - Next: {checkpoint.next}")
        
        # Show what was added at this checkpoint
        if checkpoint.values.get('risk_assessment') and i > 1:
            print(f"  - Risk assessment added")
        if checkpoint.values.get('loan_decision'):
            print(f"  - Loan decision added")
        if checkpoint.values.get('officer_notes'):
            print(f"  - Officer notes: {checkpoint.values['officer_notes'][:50]}...")
        print()
    
    print("✅ State inspection and history demonstration complete\n")


# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("LAB 1 EXERCISE SOLUTIONS")
    print("="*80)
    
    # Load environment variables from .env
    load_dotenv()

    if "GROQ_API_KEY" not in os.environ:
        os.environ["GROQ_API_KEY"] = getpass("Enter your GROQ API Key: ")

    print("✅ API Key configured")

    # Run all solutions
    solution_1_interrupt_after()
    solution_2_multiple_interrupts()
    solution_3_rejection_handling()
    solution_4_risk_scenarios()
    solution_5_state_inspection()
    
    print("\n" + "="*80)
    print("✅ ALL LAB 1 SOLUTIONS COMPLETED")
    print("="*80)
